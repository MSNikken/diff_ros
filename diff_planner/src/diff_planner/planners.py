import time
from functools import partial, partialmethod
from typing import TypeVar

import einops
import numpy as np
import torch

import diffuser.datasets.reward
import rospy
import actionlib
import tf.transformations
import pypose as pp
from diff_planner.config import load_diff_model
from diffuser.utils import apply_dict
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from diff_planner_msgs.msg import PlanPathAction


class BaseDiffusionPlanner(object):
    def __init__(self, horizon, dt_plan, state_dim, action_dim, device, t_start=None, replan_every=None):
        self.device = device
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt_plan = dt_plan
        self.plan = torch.empty((horizon, state_dim), device=device)
        self.action = torch.empty((horizon, action_dim), device=device)
        self.plan_t_start = t_start if t_start is not None else time.time_ns()

        self.obs_indices = set()
        self.latest_obs = torch.empty(state_dim, device=device)
        self.replan_every = replan_every if replan_every is not None else horizon

        self.last_replan_index = 0
        self.last_replan_time = self.plan_t_start

    def add_observation(self, obs, t=None):
        obs_index = self.get_index(t)
        if obs_index >= self.horizon - 1:  # Final state is always goal state
            return
        self.plan[obs_index] = obs
        self.obs_indices.add(obs_index)
        self.latest_obs = obs

    def generate_plan(self, start, end, start_time=None):
        self.plan = torch.empty((self.horizon, self.state_dim), device=self.device)
        self.action = torch.empty((self.horizon, self.action_dim), device=self.device)
        self.plan[-1] = end
        self.obs_indices = {-1}

        self.set_plan_start(start_time)
        self.add_observation(start, self.plan_t_start)

        self.update_plan()

    def replan(self, start_index, t=None):
        self.last_replan_index = start_index  # Before updating plan to correctly shift conditions
        self.update_plan(start_index=start_index)
        self.last_replan_time = t if t is not None else time.time_ns()

    def update_plan(self, start_index=0):
        raise NotImplementedError

    def get_setpoint(self, t=None, interpolate=None):
        action_index, remainder = self.get_index(t=t, remainder=True)
        final = action_index >= self.horizon - 1
        print(f'Getting action {action_index}')
        if final:
            print(f'Final action t={t}, tstart={self.plan_t_start}')
            return self.action[-1], True

        if action_index > self.replan_every and action_index - self.last_replan_index >= self.replan_every:
            self.replan(action_index, t=t)

        if interpolate is None:
            setpoint = self.action[action_index],
        else:
            setpoint = interpolate(self.action[action_index], self.action[action_index + 1], remainder)
        return setpoint, final

    def get_index(self, t=None, remainder=False):
        if t is None:
            t = time.time_ns()
        index_unrounded = self.last_replan_index + (t - self.last_replan_time) / (self.dt_plan * 1e9)
        index = int(index_unrounded)
        if remainder:
            decimal = index_unrounded - index
            return index, decimal
        return index

    def set_plan_start(self, t=None):
        self.plan_t_start = t if t is not None else time.time_ns()
        self.last_replan_time = self.plan_t_start
        self.last_replan_index = 0


class MockPlanner(BaseDiffusionPlanner):
    def update_plan(self, start_index=0):
        self.plan[...] = self.plan[-1]
        self.action[...] = self.plan[-1]


class GausInvDynPlanner(BaseDiffusionPlanner):
    def __init__(self, model, horizon, dt_plan, state_dim, action_dim, device, mode='pos', min_horizon=0, rew_fn=None,
                 n_samples=1, returns=None, **kwargs):
        super().__init__(horizon, dt_plan, state_dim, action_dim, device, **kwargs)
        if mode not in ['pos', 'vel']:
            raise AttributeError('Invalid action mode.')
        self.mode = mode
        self.model = model
        self.min_horizon = min_horizon
        self.rew_fn = rew_fn
        self.n_samples = n_samples
        self.returns = returns

    def update_plan(self, start_index=0):
        self.sample_plan(self.horizon - start_index, n_samples=self.n_samples, returns=self.returns)
        if self.mode == 'pos':
            self.action[0:-1] = self.plan[1:, :7]
            self.action[-1] = self.plan[-1, :7]
        else:
            raise NotImplementedError
            # self.action[now:-self.sample_step_index] = self.plan[now + 1:, 7:13]
            # self.action[-self.sample_step_index:] = self.plan[-1, :7]

    def sample_plan(self, horizon, n_samples=1, returns=None):
        print('Starting sample:')
        print('Normalizing...')

        # Due to implementation of downsampling in Unet, horizon should be a multiple of 4
        # For performance, its minimum length is settable. Additional states are placed at the
        # end as repetitions of the goal
        add_states = 0
        if horizon < self.min_horizon:
            add_states = self.min_horizon - horizon
        add_states += (4 - (horizon + add_states) % 4) % 4  # Find next multiple of 4
        horizon += add_states

        norm_plan = self.model.normalizer.normalize(self.plan, 'observations')
        print('Getting conditions...')
        # conditions = {i-self.last_replan_index: norm_plan[i][None, :] for i in self.obs_indices}
        conditions = {0: self.model.normalizer.normalize(self.latest_obs, 'observations')[None, :],
                      -1: norm_plan[-1][None, :]}

        # Place additional states
        for i in range(add_states):
            conditions[-i - 2] = norm_plan[-1][None, :]
        print(f'Condition indices: {conditions.keys()}')
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'b d -> (repeat b) d', repeat=n_samples,
        )

        if self.model.returns_condition and returns is not None:
            returns = torch.ones(n_samples, 1, device=self.device) * returns
        else:
            returns = None

        print('Forward pass...')
        norm_plan = self.model.conditional_sample(conditions, horizon=horizon, returns=returns)
        print('Unnormalizing...')
        unnorm_plan = self.model.normalizer.unnormalize(norm_plan, 'observations')[:, :horizon]
        best_plan_index = best_plan(unnorm_plan, self.rew_fn)
        self.plan[-horizon:] = unnorm_plan[best_plan_index]


DiffPlanner = TypeVar('DiffPlanner', bound=BaseDiffusionPlanner)


def pose_ros2pytorch(pose_stamped: PoseStamped, **kwargs):
    pose = torch.empty((7,), **kwargs)

    pose[0] = pose_stamped.pose.position.x
    pose[1] = pose_stamped.pose.position.y
    pose[2] = pose_stamped.pose.position.z

    pose[3] = pose_stamped.pose.orientation.x
    pose[4] = pose_stamped.pose.orientation.y
    pose[5] = pose_stamped.pose.orientation.z
    pose[6] = pose_stamped.pose.orientation.w
    return pose


def pose_pytorch2ros(pose: torch.Tensor, counter: int, norm=True):
    pose_stamped = PoseStamped()
    pose_stamped.header.seq = counter
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.header.frame_id = "global"

    pose_stamped.pose.position.x = pose[0]
    pose_stamped.pose.position.y = pose[1]
    pose_stamped.pose.position.z = pose[2]

    if norm:
        pose[3:] = pose[3:] / torch.linalg.vector_norm(pose[3:])
    pose_stamped.pose.orientation.x = pose[3]
    pose_stamped.pose.orientation.y = pose[4]
    pose_stamped.pose.orientation.z = pose[5]
    pose_stamped.pose.orientation.w = pose[6]
    return pose_stamped


def pose_franka2pytorch(msg, **kwargs):
    quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
    quaternion = quaternion / np.linalg.norm(quaternion)

    pose = torch.empty((7,), **kwargs)
    pose[0] = msg.O_T_EE[12]
    pose[1] = msg.O_T_EE[13]
    pose[2] = msg.O_T_EE[14]
    pose[3] = quaternion[0]
    pose[4] = quaternion[1]
    pose[5] = quaternion[2]
    pose[6] = quaternion[3]
    return pose


def interpolate_poses(pose1, pose2, ratio):
    interp = torch.empty((7,), device=pose1.device)
    interp[0:3] = pose1[0:3] + (pose2[0:3] - pose1[0:3]) * ratio
    rot1 = pp.SO3(pose1[3:] / torch.linalg.vector_norm(pose1[3:]))
    rot2 = pp.SO3(pose2[3:] / torch.linalg.vector_norm(pose2[3:]))
    interp[3:] = (rot1 + (rot2 * rot1.Inv()).Log() * ratio).tensor()
    return interp


def best_plan(unnorm_plan, fn=None):
    if fn is None:
        return 0
    rew = fn(unnorm_plan)
    print(f'Sample rewards. Mean: {rew.mean()} Min: {rew.min()} Max: {rew.max()}')
    return torch.argmax(rew)


class PosePlanner(object):
    def __init__(self, planner: DiffPlanner, interpolate=None, clamped=True, dt_publish=None,
                 safe_dist=0.05, safe_rot=np.pi/18):
        self.planner = planner
        self.dt_plan = planner.dt_plan
        self.dt_publish = self.dt_plan if dt_publish is None else dt_publish
        self.dt_sample = self.dt_plan if not hasattr(planner, 'dt_sample') else planner.dt_sample

        self.sub = rospy.Subscriber('franka_state_controller/franka_states', FrankaState, self.observation_cb)
        self.pub = rospy.Publisher('setpoint', PoseStamped, queue_size=1)
        self.pub_updates = rospy.Publisher('setpoint_updates', PoseStamped, queue_size=1)

        self.interpolate = interpolate
        self.last_pose = torch.empty((7,), device=self.planner.device)
        self.setpoint = torch.empty((7,), device=self.planner.device)
        self.goal = torch.empty((7,), device=self.planner.device)
        self.counter = 0
        self.final = False

        self.clamped = clamped
        self.safe_dist = safe_dist
        self.safe_rot = safe_rot

        self._as = actionlib.SimpleActionServer('plan', PlanPathAction,
                                                execute_cb=self.plan_cb, auto_start=False)
        self._as.start()

        self.wait_for_initial_pose()
        rospy.Timer(rospy.Duration(self.dt_publish), self.publish_cb)
        rospy.spin()

    def observation_cb(self, obs):
        pose = pose_franka2pytorch(obs, device=self.planner.device)
        self.planner.add_observation(pose)
        self.last_pose = pose

    def publish_cb(self, *args):
        setpoint = pose_pytorch2ros(self.setpoint, self.counter)

        # Safety
        if self.clamped:
            setpoint.pose.position.x = torch.clamp(setpoint.pose.position.x, min=0.3, max=0.5)
            setpoint.pose.position.y = torch.clamp(setpoint.pose.position.y, min=-0.25, max=0.25)
            setpoint.pose.position.z = torch.clamp(setpoint.pose.position.z, min=0.3, max=0.6)

        setpoint_pos = torch.tensor(
            [
                setpoint.pose.position.x, setpoint.pose.position.y, setpoint.pose.position.z
            ], device=self.last_pose.device
        )
        dist_from_current = torch.linalg.vector_norm(self.last_pose[0:3] - setpoint_pos)
        if dist_from_current > self.safe_dist:
            rospy.logwarn_throttle(5,
                                   f"Position setpoint too far from latest position: "
                                   f"{dist_from_current:.2f} > {self.safe_dist:.2f}. Not publishing")
            return

        setpoint_rot = pp.SO3(
            torch.tensor([
                setpoint.pose.orientation.x,
                setpoint.pose.orientation.y,
                setpoint.pose.orientation.z,
                setpoint.pose.orientation.w
            ], device=self.last_pose.device)
        )
        rot_from_current = torch.linalg.vector_norm(pp.Log(pp.SO3(self.last_pose[3:7]) * setpoint_rot.Inv()))
        if rot_from_current > self.safe_rot:
            rospy.logwarn_throttle(5,
                                   f"Rotation setpoint too far from latest rotation: "
                                   f"{rot_from_current:.2f} > {self.safe_rot:.2f}. Not publishing")
            return

        self.pub.publish(setpoint)
        self.counter += 1

    def plan_cb(self, goal):
        r = rospy.Rate(int(np.round(1 / self.dt_sample)))
        self.goal = pose_ros2pytorch(goal.goal_pose)

        rospy.loginfo(f'Generating plan...')
        tstart = rospy.Time().now()
        self.planner.generate_plan(self.last_pose, self.goal)
        duration = rospy.Time().now() - tstart
        rospy.loginfo(f'Finished plan generation in {(duration.secs + duration.nsecs / 1e9):.3f} seconds')

        self.final = False
        self.planner.set_plan_start()
        while not self.final and not rospy.is_shutdown():
            setpoint = self.next_pose()
            if setpoint is None:
                continue
            self.update_setpoint(setpoint)
            r.sleep()
        self._as.set_aborted()

    def update_setpoint(self, new):
        self.setpoint = new
        self.pub_updates.publish(pose_pytorch2ros(new, self.counter))

    def next_pose(self):
        if self.final:
            rospy.loginfo('Planner reached final state')
            return None
        setpoint, self.final = self.planner.get_setpoint(interpolate=self.interpolate)
        if setpoint is None:
            rospy.logwarn('No setpoint available')
            self.final = True
            return None
        return setpoint

    def wait_for_initial_pose(self):
        msg = rospy.wait_for_message("franka_state_controller/franka_states", FrankaState)
        self.setpoint = pose_franka2pytorch(msg)


def run_node():
    rospy.init_node('diff_planner', anonymous=False)
    config_file = rospy.get_param('~config_file', None)
    wandb_file = rospy.get_param('~wandb_file', None)
    pt_file = rospy.get_param('~pt_file', None)
    mock = rospy.get_param('~mock', False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mock:
        print('Launching mock planner.')
        planner = PosePlanner(MockPlanner(10, 0.08, 7, 7, device))
    else:
        print('Launching diffusion planner.')
        model = load_diff_model(pt_file=pt_file, config_file=config_file, wandb_path=wandb_file)
        planner = PosePlanner(
            GausInvDynPlanner(
                model, 100, 0.08, 7, 7, device,
                replan_every=30, min_horizon=32, returns=-0.01, n_samples=5,
                rew_fn=partial(
                    diffuser.datasets.reward.discounted_trajectory_rewards,
                    zones=[], discount=0.99, kin_rel_weight=0.5, kin_norm=True
                )
            ),
            interpolate=interpolate_poses,
            clamped=True,
            safe_dist=1,
            safe_rot=3.2,
            dt_publish=0.02
        )


def main():
    try:
        run_node()
    except rospy.ROSInterruptException:
        pass
