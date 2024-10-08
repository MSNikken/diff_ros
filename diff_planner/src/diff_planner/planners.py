import time
from functools import partial
from typing import TypeVar

import einops
import numpy as np
import torch

import diffuser.datasets.reward as reward
import rospy
import actionlib
import tf.transformations
import pypose as pp
from diff_planner.config import load_diff_model
from diffuser.models.guide import CostComposite, GuideManagerTrajectories
from diffuser.utils import apply_dict
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from diff_planner_msgs.msg import PlanPathAction


class BaseDiffusionPlanner(object):
    def __init__(self, horizon, dt_plan, state_dim, action_dim, device, t_start=None, replan_every=None,
                 waypoints=False):
        self.device = device
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt_plan = dt_plan
        self.plan = torch.empty((horizon, state_dim), device=device)
        self.action = torch.empty((horizon, action_dim), device=device)
        self.plan_t_start = t_start if t_start is not None else time.time_ns()
        self.waypoints = waypoints
        self.nr_setpoints = 0  # nr setpoints since replan

        self.obs_indices = set()
        self.latest_obs = torch.empty(state_dim, device=device)
        self.replan_every = replan_every if replan_every is not None else horizon

        self.last_replan_index = 0
        self.last_replan_time = self.plan_t_start

        self.last_action_index = None
        self.last_action = self.action[0]

    def add_observation(self, obs, t=None):
        self.latest_obs = obs

        # obs_index = self.get_index(t)
        # if obs_index >= self.horizon - 1:  # Final state is always goal state
        #     return
        # self.plan[obs_index] = obs
        # self.obs_indices.add(obs_index)

    def generate_plan(self, start, end, start_time=None):
        self.plan = torch.empty((self.horizon, self.state_dim), device=self.device)
        self.action = torch.empty((self.horizon, self.action_dim), device=self.device)
        self.last_action_index = None
        self.nr_setpoints = 0
        self.plan[-1] = end
        self.obs_indices = {-1}

        self.set_plan_start(start_time)
        self.add_observation(start, self.plan_t_start)

        return self.update_plan()

    def replan(self, start_index, t=None):
        self.last_replan_index = start_index  # Before updating plan to correctly shift conditions
        self.update_plan(start_index=start_index)
        self.last_replan_time = t if t is not None else time.time_ns()
        self.nr_setpoints = 0

    def update_plan(self, start_index=0):
        raise NotImplementedError

    def update_horizon(self, new_horizon):
        goal = self.plan[-1]
        if new_horizon > self.horizon:
            add_len = new_horizon - self.horizon
            self.plan = torch.cat([self.plan, torch.empty((add_len, self.state_dim), device=self.device)])
            self.action = torch.cat([self.action, torch.empty((add_len, self.state_dim), device=self.device)])
        else:
            self.plan = self.plan[:new_horizon]
            self.action = self.action[:new_horizon]
        self.plan[-1] = goal
        self.horizon = new_horizon

    def get_setpoint(self, **kwargs):
        self.nr_setpoints += 1
        if self.waypoints:
            return self.get_setpoint_from_waypoints(**kwargs)
        else:
            return self.get_setpoint_from_time_series(**kwargs)

    def get_setpoint_from_time_series(self, t=None, interpolate=None, **kwargs):
        action_index, remainder = self.get_index(t=t, remainder=True)
        final = action_index >= self.horizon - 1
        print(f'Getting action {action_index}')
        if final:
            print(f'Final action t={t}, tstart={self.plan_t_start}')
            return self.action[-1], None, True

        new_plan = (action_index > self.replan_every and action_index - self.last_replan_index >= self.replan_every
                    and torch.linalg.vector_norm(self.plan[-1][:3] - self.latest_obs[:3]) > 0.10)
        replan = None
        if new_plan:
            self.replan(action_index, t=t)
            replan = self.action

        if interpolate is None:
            setpoint = self.action[action_index],
        else:
            setpoint = interpolate(self.action[action_index], self.action[action_index + 1], remainder)
        return setpoint, replan, final

    def get_setpoint_from_waypoints(self, dist=0, interpolate=None, **kwargs):
        # Check replan
        replan = (self.nr_setpoints >= self.replan_every and
                  torch.linalg.vector_norm(self.plan[-1][:3] - self.latest_obs[:3]) > 0.10)
        if replan:
            print('Replanning')
            self.replan(start_index=self.last_action_index)
            self.last_action = self.action[self.last_action_index]
        new_plan = self.action if replan else None

        # Start new path
        if self.last_action_index is None:
            print('Starting new waypoint path')
            self.last_action_index = 0
            self.last_action = self.action[0]
            return self.action[0], new_plan, False

        next_waypoint_idx = self.last_action_index + 1
        if next_waypoint_idx >= self.horizon:
            return self.action[-1], new_plan, True

        # Calculate next setpoint
        next_dist_from_prev_wp = dist_weight_SE3(self.action[next_waypoint_idx - 1], self.last_action) + dist
        dist_to_next_wp = dist_weight_SE3(self.last_action, self.action[next_waypoint_idx])
        while dist_to_next_wp < dist:
            # Recalculate distances relative to next waypoint
            next_dist_from_prev_wp = dist - dist_to_next_wp
            dist = next_dist_from_prev_wp

            next_waypoint_idx += 1
            if next_waypoint_idx >= self.horizon:
                print('Waypoints finished')
                self.last_action_index = len(self.action) - 1
                self.last_action = self.action[-1]
                return self.action[-1], new_plan, True
            dist_to_next_wp = dist_weight_SE3(self.action[next_waypoint_idx - 1], self.action[next_waypoint_idx])
        progress_between_wp = next_dist_from_prev_wp / (next_dist_from_prev_wp + dist_to_next_wp)
        setpoint = interpolate(self.action[next_waypoint_idx - 1], self.action[next_waypoint_idx], progress_between_wp)
        self.last_action_index = next_waypoint_idx - 1
        self.last_action = setpoint
        print(f'Action towards waypoint {next_waypoint_idx}, {int(progress_between_wp * 100)}%')
        return setpoint, new_plan, False

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
                 n_samples=1, returns=None, auto=False, vel=0.005, shift=None, cost_guide=False, obstacles=None,
                 obst_cost_weight=0.1, ee_cost_weight=0.01, **kwargs):
        super().__init__(horizon, dt_plan, state_dim, action_dim, device, **kwargs)
        if mode not in ['pos', 'vel']:
            raise AttributeError('Invalid action mode.')
        self.mode = mode
        self.model = model
        self.min_horizon = min_horizon
        self.rew_fn = rew_fn
        self.n_samples = n_samples
        self.returns = returns

        # Optional cost function guidance
        self.cost_guide = cost_guide
        self.obstacles = obstacles
        self.obst_cost_weight = obst_cost_weight
        self.ee_cost_weight = ee_cost_weight

        # assert auto and rew_fn is not None or not auto
        self.auto = auto
        self.vel = vel
        if shift is None:
            shift = [0.0, 0.0, 0.0]
        self.shift = torch.tensor(shift + [0.0, 0.0, 0.0, 0.0], device=device)

    def update_plan(self, start_index=0):
        if self.auto:
            self.auto_plan(start_index)
        else:
            self.sample_plan(self.horizon - start_index, n_samples=self.n_samples, returns=self.returns)

        if self.mode == 'pos':
            self.action[0:-1] = self.plan[1:, :7]
            self.action[-1] = self.plan[-1, :7]
            return self.plan
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
        add_states += 4  # Repeated inpainting for better inpainting quality
        sample_horizon = horizon + add_states

        unnorm_plan = self.plan - self.shift
        norm_plan = self.model.normalizer.normalize(unnorm_plan, 'observations')
        print('Getting conditions...')
        # conditions = {i-self.last_replan_index: norm_plan[i][None, :] for i in self.obs_indices}
        conditions = {0: self.model.normalizer.normalize(self.latest_obs - self.shift, 'observations')[None, :],
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

        # Optional cost guidance
        guide = None
        if self.cost_guide:
            cost = CostComposite([partial(reward.cost_collision, self.obstacles), partial(reward.cost_ee, unnorm_plan[-1])],
                                 weights=[self.obst_cost_weight, self.ee_cost_weight])
            guide = GuideManagerTrajectories(cost, self.model.normalizer, clip_grad=True)


        print('Forward pass...')
        norm_plan = self.model.conditional_sample(conditions, horizon=sample_horizon, returns=returns, guide=guide)
        print('Unnormalizing...')
        unnorm_plan = self.model.normalizer.unnormalize(norm_plan, 'observations')[:, :horizon]
        best_plan_index = best_plan(unnorm_plan, self.rew_fn)
        self.plan[-horizon:] = unnorm_plan[best_plan_index] + self.shift

    def auto_plan(self, start_index):
        start = self.latest_obs
        goal = self.plan[-1]
        # return_estimate = (self.rew_fn(start[None, None, :]) + self.rew_fn(goal[None, None, :])) / 10 + 0.01
        # return_estimate = torch.clamp(return_estimate, min=-0.15, max=0)
        return_estimate = self.returns
        # horizon_estimate = max(int(torch.linalg.vector_norm(goal[:3] - start[:3])/self.vel), self.min_horizon) + start_index
        # horizon_estimate = min(max(horizon_estimate, self.min_horizon), 10*self.min_horizon)
        pregoal = self.plan[-2]
        if (torch.abs(goal[:3] - pregoal[:3]) < 0.05).all() or start_index < 5:
            horizon_estimate = self.horizon
        else:
            horizon_estimate = self.horizon + int(self.replan_every / 2)

        self.update_horizon(horizon_estimate)
        print(
            f'Auto sampling with return: {return_estimate} and horizon: {self.horizon}, horizon to go: {self.horizon - start_index}')
        self.sample_plan(self.horizon - start_index, n_samples=self.n_samples, returns=return_estimate)


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
    pose_stamped.header.frame_id = "panda_link0"

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


def path_pytorch2ros(poses, counter: int):
    path = Path()

    path.header.seq = counter
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = "panda_link0"

    for i, pose in enumerate(poses):
        path.poses.append(pose_pytorch2ros(pose, i))
    return path


def dist_weight_SE3(pose1, pose2, pos_weight=1, rot_weight=0.25):
    pos_dist = torch.linalg.vector_norm(pose2[:3] - pose1[:3], dim=-1)
    rot1 = pp.SO3(pose1[3:] / torch.linalg.vector_norm(pose1[3:]))
    rot2 = pp.SO3(pose2[3:] / torch.linalg.vector_norm(pose2[3:]))
    rot_dist = torch.linalg.vector_norm((rot2 * rot1.Inv()).Log(), dim=-1)
    return pos_weight * pos_dist + rot_weight * rot_dist


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
                 safe_dist=0.05, safe_rot=np.pi / 18, pos_only=False, wp_dist=None):
        self.planner = planner
        self.dt_plan = planner.dt_plan
        self.dt_publish = self.dt_plan if dt_publish is None else dt_publish
        self.dt_sample = self.dt_plan if not hasattr(planner, 'dt_sample') else planner.dt_sample

        self.sub = rospy.Subscriber('franka_state_controller/franka_states', FrankaState, self.observation_cb)
        # self.pub = rospy.Publisher('setpoint', PoseStamped, queue_size=1)
        self.pub = rospy.Publisher('cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=1)
        self.pub_updates = rospy.Publisher('setpoint_updates', PoseStamped, queue_size=1)
        self.pub_plan = rospy.Publisher('planned_path', Path, queue_size=1)

        self.interpolate = interpolate
        self.last_pose = torch.empty((7,), device=self.planner.device)
        self.setpoint = torch.empty((7,), device=self.planner.device)
        self.goal = torch.empty((7,), device=self.planner.device)
        self.pose_count = 0
        self.plan_count = 0
        self.final = False

        self.clamped = clamped
        self.safe_dist = safe_dist
        self.safe_rot = safe_rot
        self.wp_dist = wp_dist
        self.pos_only = pos_only

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
        setpoint = pose_pytorch2ros(self.setpoint, self.pose_count)

        # Safety
        if self.pos_only:
            setpoint.pose.orientation.x = 0.7071
            setpoint.pose.orientation.y = 0.7071
            setpoint.pose.orientation.z = 0.0
            setpoint.pose.orientation.w = 0.0

        if self.clamped:
            setpoint.pose.position.x = torch.clamp(setpoint.pose.position.x, min=0.3, max=0.5)
            setpoint.pose.position.y = torch.clamp(setpoint.pose.position.y, min=-0.25, max=0.25)
            setpoint.pose.position.z = torch.clamp(setpoint.pose.position.z, min=0.09, max=0.6)

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
        self.pose_count += 1

    def plan_cb(self, goal):
        r = rospy.Rate(int(np.round(1 / self.dt_sample)))
        self.goal = pose_ros2pytorch(goal.goal_pose)

        rospy.loginfo(f'Generating plan...')
        tstart = rospy.Time().now()
        plan = self.planner.generate_plan(self.last_pose, self.goal)
        duration = rospy.Time().now() - tstart
        rospy.loginfo(f'Finished plan generation in {(duration.secs + duration.nsecs / 1e9):.3f} seconds')
        self.pub_plan.publish(path_pytorch2ros(plan, self.plan_count))
        self.plan_count += 1

        self.final = False
        self.planner.set_plan_start()
        while not self.final and not rospy.is_shutdown():
            setpoint, plan = self.next_pose()
            if setpoint is None:
                continue
            if plan is not None:
                self.pub_plan.publish(path_pytorch2ros(plan, self.plan_count))
                self.plan_count += 1
            self.update_setpoint(setpoint)
            r.sleep()
        self._as.set_aborted()

    def update_setpoint(self, new):
        self.setpoint = new
        self.pub_updates.publish(pose_pytorch2ros(new, self.pose_count))

    def next_pose(self):
        if self.final:
            rospy.loginfo('Planner reached final state')
            return None
        setpoint, plan, self.final = self.planner.get_setpoint(interpolate=self.interpolate, dist=self.wp_dist)
        if setpoint is None:
            rospy.logwarn('No setpoint available')
            self.final = True
            return None
        return setpoint, plan

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
    print(f'Using device: {device}')
    if mock:
        print('Launching mock planner.')
        planner = PosePlanner(MockPlanner(10, 0.08, 7, 7, device))
    else:
        print('Launching diffusion planner.')
        model = load_diff_model(pt_file=pt_file, config_file=config_file, wandb_path=wandb_file)
        planner = PosePlanner(
            GausInvDynPlanner(
                model, 120, 0.08, 7, 7, device, auto=True,
                replan_every=61, min_horizon=32, returns=-0.01, n_samples=10,
                rew_fn=partial(
                    reward.discounted_trajectory_rewards,
                    zones=[reward.Zone(xmin=0.3, ymin=-0.1, zmin=0.3, xmax=0.5, ymax=0.1, zmax=0.45)],
                    # zones=[reward.Zone(xmin=0.3, ymin=-0.15, zmin=0.3, xmax=0.41, ymax=-0.05, zmax=0.6),
                    #        reward.Zone(xmin=0.39, ymin=0.05, zmin=0.3, xmax=0.5, ymax=0.15, zmax=0.6)],
                    # zones=[reward.Zone(xmin=0.3, ymin=0.0, zmin=0.3, xmax=0.5, ymax=0.25, zmax=0.4),
                    #       reward.Zone(xmin=0.3, ymin=-0.05, zmin=0.3, xmax=0.4, ymax=0.05, zmax=0.6)],
                    discount=0.99,
                    dist_scale=0.1, kin_rel_weight=0, kin_norm=True
                ),
                waypoints=True,
                cost_guide=True,
                obstacles=[reward.Zone(xmin=0.3, ymin=-0.1, zmin=0.3, xmax=0.5, ymax=0.1, zmax=0.45)],
                # shift=[0.0, 0.0, -0.25]
                shift=[0.0, 0.0, 0.0]
            ),
            interpolate=interpolate_poses,
            clamped=True,
            safe_dist=1,
            safe_rot=3.2,
            pos_only=True,
            dt_publish=0.02,
            wp_dist=2e-2,
        )


def main():
    try:
        run_node()
    except rospy.ROSInterruptException:
        pass
