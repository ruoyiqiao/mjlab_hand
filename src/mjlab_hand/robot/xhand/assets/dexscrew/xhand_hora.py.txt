# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Written by Paper Authors
# Copyright (c) 2025 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: Lessons from Learning to Spin "Pens"
# Copyright (c) 2024 All Authors
# Licensed under MIT License
# https://github.com/HaozhiQi/penspin/
# --------------------------------------------------------

import os
from typing import Optional
import torch
import omegaconf
import numpy as np
import time 

from glob import glob
from collections import OrderedDict

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_conjugate, quat_mul, to_torch, quat_apply, tensor_clamp, torch_rand_float, quat_from_euler_xyz

from ..utils.point_cloud_prep import sample_cylinder
from .base.vec_task import VecTask
from dexscrew.utils.misc import tprint
import torch.nn.functional as F


class XHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        # before calling init in VecTask, need to do
        # 0. setup the dim info
        self.numActions = config['env']['numActions']
        self.fingers_num = 5
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup rewards
        self._setup_reward_config(config['env']['reward'])
        # unclassified config
        self.base_obj_scale = config['env']['baseObjScale']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.rotation_axis = config['env']['rotation_axis']
        self.reset_z_threshold = self.config['env']['reset_z_threshold']
        self.reset_dist_threshold = self.config['env']['reset_dist_threshold']
        self.with_camera = config['env']['enableCameraSensors']
        self.nut_termination_history_len = config['env']['object']['nut_termination_history_len'] 
        self.nut_stagnation_eps = config['env']['object']['nut_stagnation_eps']
        self.screw_upper_limit = config['env']['object']['screw_upper_limit']

        # Important: map CUDA device IDs to Vulkan ones.
        graphics_device_id = 0

        super().__init__(config, sim_device, graphics_device_id, headless)
        self.eval_done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.xhand_hand_default_dof_pos = torch.zeros(self.num_xhand_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.xhand_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_xhand_hand_dofs]
        self.xhand_hand_dof_pos = self.xhand_hand_dof_state[..., 0]
        self.xhand_hand_dof_vel = self.xhand_hand_dof_state[..., 1]
        self.pre_state = torch.zeros_like(self.xhand_hand_dof_pos)
        # Disable pose-diff penalty for thumb joints by masking them out (thumb DOFs contain 'thumb' in name)
        dof_names = self.gym.get_asset_dof_names(self.hand_asset)
        thumb_indices = [i for i, name in enumerate(dof_names) if 'thumb' in name]
        self.pose_diff_penalty_mask = torch.ones(self.num_actions, device=self.device, dtype=torch.float)
        if len(thumb_indices) > 0:
            self.pose_diff_penalty_mask[thumb_indices] = 0.0
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.contact_thresh = torch.zeros((self.num_envs, self.num_contacts), dtype=torch.float, device=self.device)

        self.saved_grasping_states = {}
        num_random_poses = 5000
        if self.config['env']['initPose'] == 'nutbolt_inclined':
            joint_values = {
                "left_hand_index_bend_joint":-0.17,
                "left_hand_index_joint1":1.1,
                "left_hand_index_joint2":0.4,
                "left_hand_mid_joint1":1.1,
                "left_hand_mid_joint2":0.4,
                "left_hand_pinky_joint1":0,
                "left_hand_pinky_joint2":0,
                "left_hand_ring_joint1":0,
                "left_hand_ring_joint2":0,
                "left_hand_thumb_bend_joint":1.3,
                "left_hand_thumb_rota_joint1":0.5,
                "left_hand_thumb_rota_joint2":0.45,
                }
        elif self.config['env']['initPose'] == 'screwdriver_inclined':
            joint_values = {
                "left_hand_index_bend_joint": -0.036,
                "left_hand_index_joint1":1.15,
                "left_hand_index_joint2":0.5,
                "left_hand_mid_joint1":0.925,
                "left_hand_mid_joint2": 0.58,
                "left_hand_pinky_joint1":0,
                "left_hand_pinky_joint2":0,
                "left_hand_ring_joint1":1.3,
                "left_hand_ring_joint2":0.43,
                "left_hand_thumb_bend_joint":1.455,
                "left_hand_thumb_rota_joint1":0.817,
                "left_hand_thumb_rota_joint2":0.154,
                }

        assert self.gym.get_asset_dof_names(self.hand_asset) == list(joint_values.keys())
        self.joint_values_lst = list(joint_values.values())
        for s in self.randomize_scale_list:
            scale_key = str(s)
            random_pose_data = torch.zeros((num_random_poses, 19), device=self.device, dtype=torch.float)
            
            # Pre-defined pose
            for i in range(12):
                lower, upper = self.xhand_dof_lower_limits[i], self.xhand_dof_upper_limits[i]
                random_pose_data[:, i] = torch.ones(num_random_poses, device=self.device) * self.joint_values_lst[i]
            random_pose_data[:, 12] = torch.zeros(num_random_poses, device=self.device)  # x
            random_pose_data[:, 13] = torch.zeros(num_random_poses, device=self.device)  # y
            random_pose_data[:, 14] = self.reset_z_threshold + 0.1  # z - slightly above threshold
            random_pose_data[:, 15:18] = 0.0 
            random_pose_data[:, 18]   = 1.0
            quat_norm = torch.norm(random_pose_data[:, 15:19], dim=1, keepdim=True)
            random_pose_data[:, 15:19] = random_pose_data[:, 15:19] / quat_norm
            
            self.saved_grasping_states[scale_key] = random_pose_data
            print(f"Generated {num_random_poses} random initial poses for XHand at scale {s}")


        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        sign, axis = self.rotation_axis[0], self.rotation_axis[1]
        axis_index = ['x', 'y', 'z'].index(axis)
        self.rot_axis_buf[:, axis_index] = 1
        self.rot_axis_buf[:, axis_index] = -self.rot_axis_buf[:, axis_index] if sign == '-' else self.rot_axis_buf[:, axis_index]

        # useful buffers
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) # +1 for screw
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        # there is an extra dim [self.control_freq_inv] because we want to get a mean over multiple control steps
        self.torques = torch.zeros((self.num_envs, self.control_freq_inv, self.num_dofs), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.control_freq_inv, self.num_actions), device=self.device, dtype=torch.float)

        # calculate velocity at control frequency instead of simulated frequency
        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()
        self.ft_pos_prev = self.fingertip_pos.clone()
        self.ft_rot_prev = self.fingertip_orientation.clone()
        self.dof_vel_prev = self.dof_vel_finite_diff.clone()
        self.nut_dof_pos_prev = self.nut_dof_pos.clone()

        self.obj_linvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_angvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.ft_linvel_at_cf = torch.zeros((self.num_envs, 4 * 3), device=self.device, dtype=torch.float)
        self.ft_angvel_at_cf = torch.zeros((self.num_envs, 4 * 3), device=self.device, dtype=torch.float)
        self.nut_dof_vel_cf = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.dof_acc = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)

        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

        # debug and statistics
        self.evaluate = self.config['on_evaluation']
        self.evaluate_cache_name = self.config['eval_cache_name']
        self.stat_sum_rewards = [0 for _ in self.object_type_list]  # all episode reward
        self.stat_sum_episode_length = [0 for _ in self.object_type_list]  # average episode length
        self.stat_sum_rotate_rewards = [0 for _ in self.object_type_list]  # rotate reward, with clipping
        self.stat_sum_rotate_penalty = [0 for _ in self.object_type_list]  # rotate penalty with clipping
        self.stat_sum_unclip_rotate_rewards = [0 for _ in self.object_type_list]  # rotate reward, with clipping
        self.stat_sum_unclip_rotate_penalty = [0 for _ in self.object_type_list]  # rotate penalty with clipping
        self.extrin_log = []
        self.env_evaluated = [0 for _ in self.object_type_list]
        self.evaluate_iter = 0

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self._create_object_asset()
        xhand_hand_dof_props = self._parse_hand_dof_props()
        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_xhand_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_xhand_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_xhand_hand_bodies + 2
        max_agg_shapes = self.num_xhand_hand_shapes + 2

        self.envs = []
        self.vid_record_tensor = None  # Used for record video during training, NOT FOR POLICY OBSERVATION
        self.object_init_state = []

        self.hand_indices = []
        self.hand_actors = []
        self.object_indices = []
        self.object_type_at_env = []

        self.obj_point_clouds = []

        xhand_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.screw_base_rb_handle = xhand_hand_rb_count + 0  # base
        self.screw_bolt_rb_handle = xhand_hand_rb_count + 1  # bolt  
        self.screw_nut_rb_handle = xhand_hand_rb_count + 2   # nut
        object_rb_count = 3
        
        self.object_rb_handles = list(range(xhand_hand_rb_count, xhand_hand_rb_count + object_rb_count))

        for i in range(num_envs):
            tprint(f'{i} / {num_envs}')
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 1)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, xhand_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_actors.append(hand_actor)

            # add object
            eval_object_type = self.config['env']['object']['evalObjectType']
            if eval_object_type is None:
                object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            else:
                object_type_id = self.object_type_list.index(eval_object_type)

            self.object_type_at_env.append(object_type_id)
            object_asset = self.object_asset_list[object_type_id]
            print(f"env {i} object_asset id: {object_type_id}")

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 2)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self._update_priv_buf(env_id=i, name='screw_joint_friction', value=0.2)

            self.obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                self.obj_scale = np.random.uniform(
                    self.randomize_scale_list[i % num_scales] - 0.025,
                    self.randomize_scale_list[i % num_scales] + 0.025
                )
            self.gym.set_actor_scale(env_ptr, object_handle, self.obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=self.obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                if 'screw' in self.config['env']['object']['type']:
                    assert len(prop) == 3, "Screw object should have 2 rigid bodies"
                    # Base COM randomization (smaller range since it's the stabilizing part)
                    base_com = [
                        np.random.uniform(self.randomize_com_lower/2, self.randomize_com_upper/2),
                        np.random.uniform(self.randomize_com_lower/2, self.randomize_com_upper/2),
                        np.random.uniform(self.randomize_com_lower/2, self.randomize_com_upper/2)
                    ]

                    nut_com = [
                        np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                        np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                        0.35 + np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)
                    ]

                    prop[0].com.x, prop[0].com.y, prop[0].com.z = base_com
                    prop[2].com.x, prop[2].com.y, prop[2].com.z = nut_com
                    obj_com = base_com
                    ro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
                else:
                    assert len(prop) == 1
                    obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                            np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                            np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                    prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            obj_restitution = 0.0
            # Friction randomization
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                obj_restitution = np.random.uniform(0, 1)

                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)
            self._update_priv_buf(env_id=i, name='obj_restitution', value=obj_restitution)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.point_cloud_sampled_dim > 0:
                self.obj_point_clouds.append(self.asset_point_clouds[object_type_id] * self.obj_scale)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # for training record, visualized in tensorboard
            if self.with_camera:
                self.vid_record_tensor = self._create_camera(env_ptr)

            self.envs.append(env_ptr)

        self.obj_point_clouds = to_torch(np.array(self.obj_point_clouds), device=self.device, dtype=torch.float)
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.object_type_at_env = to_torch(self.object_type_at_env, dtype=torch.long, device=self.device)

    def _create_camera(self, env_ptr) -> torch.Tensor:
        """Create a camera in a particular environment. Should be called in _create_envs."""
        camera_props = gymapi.CameraProperties()
        camera_props.width = 256
        camera_props.height = 256
        camera_props.enable_tensors = True
        
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

        cam_pos = gymapi.Vec3(0.0, 0.2, 0.75)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)

        self.gym.set_camera_location(camera_handle, env_ptr, cam_pos, cam_target)
        # obtain camera tensor
        vid_record_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
        )
        # wrap camera tensor in a pytorch tensor
        vid_record_tensor.device = 0
        torch_vid_record_tensor = gymtorch.wrap_tensor(vid_record_tensor)
        assert torch_vid_record_tensor.shape == (camera_props.height, camera_props.width, 4)

        return torch_vid_record_tensor

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            dof_names = self.gym.get_asset_dof_names(self.hand_asset)
            pgain_priv = torch.zeros((len(env_ids), self.num_xhand_hand_dofs), device=self.device, dtype=torch.float)
            dgain_priv = torch.zeros((len(env_ids), self.num_xhand_hand_dofs), device=self.device, dtype=torch.float)
            for joint_idx, joint_name in enumerate(dof_names):
                p_lower = self.randomize_p_gain_lower 
                p_upper = self.randomize_p_gain_upper
                d_lower = self.randomize_d_gain_lower
                d_upper = self.randomize_d_gain_upper
                pgain_priv[:, joint_idx] = torch_rand_float(
                    p_lower, p_upper, (len(env_ids), 1), device=self.device).squeeze(-1)
                dgain_priv[:, joint_idx] = torch_rand_float(
                    d_lower, d_upper, (len(env_ids), 1), device=self.device).squeeze(-1)

                self.p_gain[env_ids, joint_idx] = pgain_priv[:, joint_idx]
                self.d_gain[env_ids, joint_idx] = dgain_priv[:, joint_idx]

            self._update_priv_buf(env_ids, 'pgain', pgain_priv)
            self._update_priv_buf(env_ids, 'dgain', dgain_priv)

        self.random_obs_noise_e[env_ids] = torch.normal(0, self.random_obs_noise_e_scale, size=(len(env_ids), self.num_actions), device=self.device, dtype=torch.float)
        self.random_action_noise_e[env_ids] = torch.normal(0, self.random_action_noise_e_scale, size=(len(env_ids), self.num_actions), device=self.device, dtype=torch.float)
        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            # single object (category) case:
            sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
            if self.config['env']['initPose'] == 'nutbolt_inclined':
                random_noise_x, random_noise_y = np.random.uniform(0, 0.0075), np.random.uniform(0, 0.0075)
                self.object_x, self.object_y, self.object_z = 0.0175 + random_noise_x, 0.06 + random_noise_y, 0
            elif self.config['env']['initPose'] == 'screwdriver_inclined':
                random_noise_x, random_noise_y = np.random.uniform(0, 0.0075), np.random.uniform(0, 0.0075)
                self.object_x, self.object_y, self.object_z = 0.009 + random_noise_x, 0.06 + random_noise_y, 0

            sampled_pose[:, self.numActions + 0] = self.object_x    # X (left-right: postive-nagative)
            sampled_pose[:, self.numActions + 1] = self.object_y  # Y (forward-backward: nagative-positive)
            sampled_pose[:, self.numActions + 2] = self.object_z   # Z (above palm, so it can fall downward)
            
            object_pose_noise = torch.normal(0, self.random_pose_noise, size=(sampled_pose.shape[0], 7), device=self.device, dtype=torch.float)
            object_pose_noise[:, 3:] = 0
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, self.numActions:] + object_pose_noise
            
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
            pos = sampled_pose[:, :self.numActions]
            self._update_priv_buf(env_id=s_ids, name='hand_joint_pos', value=pos) # give the policy correct hand joint position
            half_widths = torch.tensor(0.1 * (self.xhand_dof_upper_limits - self.xhand_dof_lower_limits) / 2.0, device=self.device, dtype=torch.float)
            hand_joint_noise = (torch.rand_like(half_widths) * 2 - 1) * half_widths
            pos = pos + hand_joint_noise
            self.sampled_pose = sampled_pose[:, :self.numActions]
            self.xhand_hand_dof_pos[s_ids, :] = pos
            self.xhand_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_xhand_hand_dofs] = pos
            self.cur_targets[s_ids, :self.num_xhand_hand_dofs] = pos
            
            screw_joint_init = torch.zeros((len(s_ids), self.num_dofs - self.num_xhand_hand_dofs), device=self.device)
            self.prev_targets[s_ids, self.num_xhand_hand_dofs:] = screw_joint_init
            self.cur_targets[s_ids, self.num_xhand_hand_dofs:] = screw_joint_init
            full_dof_state = torch.cat([pos, screw_joint_init], dim=1)
            self.init_pose_buf[s_ids, :] = full_dof_state

        # X-axis rotation
        random_degrees_x = torch.rand(len(env_ids), device=self.device) * 10 + 20 # randomize orientation bewteen 20-30 degrees
        angles_x = np.pi/2 - random_degrees_x * (np.pi / 180)
        cos_half_x, sin_half_x = torch.cos(angles_x / 2), torch.sin(angles_x / 2)
        
        # Y-axis rotation
        random_degrees_y = torch.rand(len(env_ids), device=self.device) * 10 - 5 + 90 # randomize orientation bewteen -5-5 degrees, 90 is the correction term for y
        angles_y = np.pi/2 - random_degrees_y * (np.pi / 180)
        cos_half_y, sin_half_y = torch.cos(angles_y / 2), torch.sin(angles_y / 2)
        
        # Create quaternions for x and y rotations
        quat_x = torch.stack([sin_half_x, torch.zeros_like(sin_half_x), torch.zeros_like(sin_half_x), cos_half_x], dim=1)
        quat_y = torch.stack([torch.zeros_like(sin_half_y), sin_half_y, torch.zeros_like(sin_half_y), cos_half_y], dim=1)
        
        quats = torch.stack([
            quat_y[:, 3] * quat_x[:, 0] + quat_y[:, 0] * quat_x[:, 3] + quat_y[:, 1] * quat_x[:, 2] - quat_y[:, 2] * quat_x[:, 1],  # x
            quat_y[:, 3] * quat_x[:, 1] - quat_y[:, 0] * quat_x[:, 2] + quat_y[:, 1] * quat_x[:, 3] + quat_y[:, 2] * quat_x[:, 0],  # y
            quat_y[:, 3] * quat_x[:, 2] + quat_y[:, 0] * quat_x[:, 1] - quat_y[:, 1] * quat_x[:, 0] + quat_y[:, 2] * quat_x[:, 3],  # z
            quat_y[:, 3] * quat_x[:, 3] - quat_y[:, 0] * quat_x[:, 0] - quat_y[:, 1] * quat_x[:, 1] - quat_y[:, 2] * quat_x[:, 2]   # w
        ], dim=1)
        
        hand_env_indices = self.hand_indices[env_ids]
        self.root_state_tensor[hand_env_indices, 3:7] = quats
        self._update_priv_buf(env_id=env_ids, name='hand_orientation', value=quats)

        # position randomization
        original_pos = torch.zeros_like(self.root_state_tensor[hand_env_indices, :3])
        original_pos[:, 2] = 0.21
        pos = original_pos + torch.rand_like(original_pos) * 0.001 
        self.root_state_tensor[hand_env_indices, :3] = pos
        self._update_priv_buf(env_id=env_ids, name='hand_position', value=pos)

        # Object inclination randomization (±5 degrees tilt from vertical Z-axis)
        if self.object_tilt_enabled:
            tilt_angle = torch.rand(len(env_ids), device=self.device) * 5.0 * (np.pi / 180)  # 0-5 degrees in radians
            tilt_direction = torch.rand(len(env_ids), device=self.device) * 2 * np.pi
            
            # Convert to tilt around X and Y axes
            x_rotation = tilt_angle * torch.sin(tilt_direction)
            y_rotation = tilt_angle * torch.cos(tilt_direction)
            
            # Create quaternions for small rotations (small angle approximation)
            cos_half_x, sin_half_x = torch.cos(x_rotation / 2), torch.sin(x_rotation / 2)
            cos_half_y, sin_half_y = torch.cos(y_rotation / 2), torch.sin(y_rotation / 2)
            
            quat_x_obj = torch.stack([sin_half_x, torch.zeros_like(sin_half_x), torch.zeros_like(sin_half_x), cos_half_x], dim=1)
            quat_y_obj = torch.stack([torch.zeros_like(sin_half_y), sin_half_y, torch.zeros_like(sin_half_y), cos_half_y], dim=1)
            
            # Combine rotations by multiplying quaternions (quat_y * quat_x)
            object_quats = torch.stack([
                quat_y_obj[:, 3] * quat_x_obj[:, 0] + quat_y_obj[:, 0] * quat_x_obj[:, 3] + quat_y_obj[:, 1] * quat_x_obj[:, 2] - quat_y_obj[:, 2] * quat_x_obj[:, 1],  # x
                quat_y_obj[:, 3] * quat_x_obj[:, 1] - quat_y_obj[:, 0] * quat_x_obj[:, 2] + quat_y_obj[:, 1] * quat_x_obj[:, 3] + quat_y_obj[:, 2] * quat_x_obj[:, 0],  # y
                quat_y_obj[:, 3] * quat_x_obj[:, 2] + quat_y_obj[:, 0] * quat_x_obj[:, 1] - quat_y_obj[:, 1] * quat_x_obj[:, 0] + quat_y_obj[:, 2] * quat_x_obj[:, 3],  # z
                quat_y_obj[:, 3] * quat_x_obj[:, 3] - quat_y_obj[:, 0] * quat_x_obj[:, 0] - quat_y_obj[:, 1] * quat_x_obj[:, 1] - quat_y_obj[:, 2] * quat_x_obj[:, 2]   # w
            ], dim=1)
            
            # Apply to object
            object_env_indices = self.object_indices[env_ids]
            self.root_state_tensor[object_env_indices, 3:7] = object_quats
            self._update_priv_buf(env_id=env_ids, name='obj_orientation', value=object_quats)
            
            # Compensate Z position for tilt to prevent collision with ground
            shaft_radius = 0.00625  # From URDF: shaft radius = 0.00625m
            z_compensation = shaft_radius * torch.tan(tilt_angle)
            current_z = self.root_state_tensor[object_env_indices, 2]
            self.root_state_tensor[object_env_indices, 2] = current_z + z_compensation

        else:
            # No tilt - set object to perfectly vertical orientation (identity quaternion)
            object_env_indices = self.object_indices[env_ids]
            identity_quats = torch.zeros((len(env_ids), 4), device=self.device, dtype=torch.float)
            identity_quats[:, 3] = 1.0  # w = 1, x = y = z = 0 (identity quaternion)
            
            self.root_state_tensor[object_env_indices, 3:7] = identity_quats
            self._update_priv_buf(env_id=env_ids, name='obj_orientation', value=identity_quats)
        
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_actor_indices = torch.cat([object_indices, hand_indices])
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(all_actor_indices), len(all_actor_indices))
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # reset tactile
        self.contact_thresh[env_ids] = 0.05

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.dof_vel_finite_diff[:] = 0
        self.nut_dof_pos_history[env_ids] = 0
        self.nut_contact_history[env_ids] = 0
        
        self.at_reset_buf[env_ids] = 1
        
    def compute_observations(self): 
        self._refresh_gym()
        # observation noise
        random_obs_noise_t = torch.normal(0, self.random_obs_noise_t_scale, size=self.xhand_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noisy_joint_pos = random_obs_noise_t + self.random_obs_noise_e + self.xhand_hand_dof_pos 

        t_buf = (self.obs_buf_lag_history[:, -3:,:self.obs_buf.shape[1]//3].reshape(self.num_envs, -1)).clone()
        self.obs_buf[:, :t_buf.shape[1]] = t_buf  # [1, 96]
        
        # deal with normal observation, do sliding windows
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # xhand dim [1, 1, 12]
        cur_tar_buf = self.cur_targets[:, None, :self.num_actions]  # [1, 1, 12]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # [1, 1, 24]

        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1) # torch.Size([48, 80, 24])

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:self.numActions] = self.init_pose_buf[at_reset_env_ids, :self.num_actions].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, self.numActions:self.numActions*2] = self.init_pose_buf[at_reset_env_ids, :self.num_actions].unsqueeze(1)
        
        # velocity reset
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]
        self.nut_dof_vel_cf[at_reset_env_ids] = self.nut_dof_vel[at_reset_env_ids]
        
        self.at_reset_buf[at_reset_env_ids] = 0
        rand_rpy = torch.normal(0, self.noisy_rpy_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float)
        rand_quat = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        _noisy_quat = quat_mul(rand_quat, self.object_rot)
        _noisy_position = torch.normal(0, self.noisy_pos_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float) + self.object_pos
        
        # Update nut history buffers for termination conditions
        prev_nut_dof_pos_history = self.nut_dof_pos_history[:, 1:].clone()
        cur_nut_dof_pos = self.nut_dof_pos.clone().unsqueeze(1)
        self.nut_dof_pos_history[:] = torch.cat([prev_nut_dof_pos_history, cur_nut_dof_pos], dim=1)
            
        prev_nut_contact_history = self.nut_contact_history[:, 1:].clone()
        cur_nut_contact = self.nut_contact.clone().unsqueeze(1)
        self.nut_contact_history[:] = torch.cat([prev_nut_contact_history, cur_nut_contact], dim=1)
        
        if len(at_reset_env_ids) > 0:
            self.nut_dof_pos_history[at_reset_env_ids] = self.nut_dof_pos[at_reset_env_ids].unsqueeze(1).repeat(1, self.nut_termination_history_len, 1)
            self.nut_contact_history[at_reset_env_ids] = self.nut_contact[at_reset_env_ids].unsqueeze(1).repeat(1, self.nut_termination_history_len, 1)
        
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :self.numActions*2]  # [1, 30, 32]
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_orientation', value=self.object_rot.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_linvel', value=self.obj_linvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_orientation', value=self.fingertip_orientation.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_linvel', value=self.ft_linvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_angvel', value=self.ft_angvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='nut_pos', value=self.nut_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='nut_dof_pos', value=self.nut_dof_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='nut_dof_vel', value=self.nut_dof_vel_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_position', value=self.fingertip_pos.clone())

        if self.point_cloud_sampled_dim > 0:
            # for collecting bc data
            self.point_cloud_buf[:, :self.point_cloud_sampled_dim] = quat_apply(
                self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1), self.obj_point_clouds
            ) + self.object_pos[:, None]  # [1, 100, 3]
    
    def _get_reward_scale_by_name(self, name):
        env_steps = (self.gym.get_frame_count(self.sim) * len(self.envs))
        agent_steps = env_steps // self.control_freq_inv
        init_scale, final_scale, curr_start, curr_end = self.reward_scale_dict[name]
        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            # discretize to [0, 0.05, 1.0] instead of continuous value
            # during batch collection, avoid reward confusion
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1
        if self.evaluate:
            curr_progress = 1
        return init_scale + (final_scale - init_scale) * curr_progress
    
    def _get_current_angvel_penalty_threshold(self):
        """Get the current angular velocity penalty threshold based on curriculum progress."""
        env_steps = (self.gym.get_frame_count(self.sim) * len(self.envs))
        agent_steps = env_steps // self.control_freq_inv
        init_threshold, final_threshold, curr_start, curr_end = self.angvel_penalty_threshold_curriculum
        
        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            # discretize to avoid confusion during batch collection
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1
            
        if self.evaluate:
            curr_progress = 1
            
        current_threshold = init_threshold + (final_threshold - init_threshold) * curr_progress
        return current_threshold

    def compute_reward(self, actions):
        # Update current angular velocity penalty threshold based on curriculum
        current_angvel_penalty_threshold = self._get_current_angvel_penalty_threshold()
        
        # pose diff penalty (thumb DOFs masked out)
        pose_diff_penalty = (
            ((self.xhand_hand_dof_pos - self.init_pose_buf[..., :self.num_actions]) ** 2)
            * self.pose_diff_penalty_mask
        ).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques[:, -1, :self.num_actions] ** 2).sum(-1)
        work_penalty = (((torch.abs(self.torques[:, -1, :self.num_actions]) * torch.abs(self.dof_vel_finite_diff[:, -1])).sum(-1)) ** 2)

        angdiff = self.quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)

        # Calculate proximity reward
        nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        nut_pos = nut_states[..., :3]
        thumb_pos, index_pos = self.rigid_body_states[:, self.fingertip_handles[-1], :3], self.rigid_body_states[:, self.fingertip_handles[0], :3]
        thumb_dist, index_dist = torch.norm(thumb_pos - nut_pos, dim=-1), torch.norm(index_pos - nut_pos, dim=-1)
        mean_dist = 0.5 * (thumb_dist + index_dist)
        ratio = mean_dist / self.reset_dist_threshold
        proximity_reward = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

        nut_dof_linvel = ((self.nut_dof_pos.squeeze(-1) - self.nut_dof_pos_prev.squeeze(-1)) / (self.control_freq_inv * self.dt)).squeeze(-1)
        self.nut_dof_vel_cf = nut_dof_linvel.unsqueeze(-1)
        rotate_reward_raw = torch.clip(nut_dof_linvel, max=self.angvel_clip_max, min=self.angvel_clip_min)
        rotate_reward = rotate_reward_raw

        rotate_penalty_raw = torch.where(nut_dof_linvel > current_angvel_penalty_threshold, nut_dof_linvel - current_angvel_penalty_threshold, 0)
        reverse_penalty = torch.where(nut_dof_linvel < 0, torch.abs(nut_dof_linvel) * 2.0, torch.zeros_like(nut_dof_linvel))
        rotate_penalty = rotate_penalty_raw

        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        self.obj_angvel_at_cf = object_angvel
        self.obj_linvel_at_cf = object_linvel
        ft_angdiff = self.quat_to_axis_angle(quat_mul(self.fingertip_orientation.reshape(-1, 4), quat_conjugate(self.ft_rot_prev.reshape(-1, 4)))).reshape(-1, self.fingers_num*3)
        self.ft_angvel_at_cf = ft_angdiff / (self.control_freq_inv * self.dt)
        self.ft_linvel_at_cf = ((self.fingertip_pos - self.ft_pos_prev) / (self.control_freq_inv * self.dt))
        self.z_dist_penalty = (self.object_pos[:, 2] - self.object_z) ** 2

        if self.point_cloud_sampled_dim > 0:
            point_cloud_z = self.point_cloud_buf[:, :self.point_cloud_sampled_dim, -1]
            z_dist_penalty = point_cloud_z.max(axis=1)[0] - point_cloud_z.min(axis=1)[0] 
            z_dist_penalty[z_dist_penalty <= 0.03] = 0
        else:
            z_dist_penalty = to_torch([0], device=self.device)

        self.rew_buf[:] = compute_hand_reward(
            rotate_reward, self._get_reward_scale_by_name('rotate_reward'),
            pose_diff_penalty, self._get_reward_scale_by_name('pose_diff_penalty'),
            torque_penalty, self._get_reward_scale_by_name('torque_penalty'),
            work_penalty, self._get_reward_scale_by_name('work_penalty'),
            z_dist_penalty, self._get_reward_scale_by_name('pc_z_dist_penalty'),
            rotate_penalty, self._get_reward_scale_by_name('rotate_penalty'),
            proximity_reward, self._get_reward_scale_by_name('proximity_reward')
        )
        

        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras['step_all_reward'] = self.rew_buf.mean()
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = torch.abs(object_angvel[:, 0]).mean()
        self.extras['pitch'] = torch.abs(object_angvel[:, 1]).mean()
        self.extras['yaw'] = torch.abs(object_angvel[:, 2]).mean()
        self.extras['z_dist_penalty'] = z_dist_penalty.mean()
        self.extras['rotate_penalty'] = rotate_penalty.mean()
        
        # curriculum tracking
        self.extras['curriculum/angvel_penalty_threshold'] = current_angvel_penalty_threshold

        # screw-specific metrics
        self.extras['screw/angular_velocity'] = self.nut_dof_vel.mean()
        self.extras['screw/angular_position'] = self.nut_dof_pos.mean()
        self.extras['screw/positive_vel_ratio'] = (self.nut_dof_vel > 0).float().mean()

        if self.evaluate:
            vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
            for i in range(len(self.object_type_list)):
                env_ids = torch.where(self.object_type_at_env == i)
                if len(env_ids[0]) > 0:
                    running_mask = 1 - self.eval_done_buf[env_ids]
                    self.stat_sum_rewards[i] += (running_mask * self.rew_buf[env_ids]).sum()
                    self.stat_sum_episode_length[i] += running_mask.sum()
                    self.stat_sum_rotate_rewards[i] += (running_mask * rotate_reward[env_ids]).sum()
                    self.stat_sum_unclip_rotate_rewards[i] += (running_mask * vec_dot[env_ids]).sum()

                    if self.config['env']['object']['evalObjectType'] is not None:
                        flip = running_mask * self.reset_buf[env_ids]
                        self.env_evaluated[i] += flip.sum()
                        self.eval_done_buf[env_ids] += flip

                    info = f'Progress: {self.evaluate_iter} / {self.max_episode_length}'
                    tprint(info)
            self.evaluate_iter += 1

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        targets = self.prev_targets + self.action_scale * self.actions
        self.cur_targets[:, :self.num_xhand_hand_dofs] = tensor_clamp(targets[:, :self.num_xhand_hand_dofs], self.xhand_hand_dof_lower_limits, self.xhand_hand_dof_upper_limits)

        # get prev* buffer here
        self.prev_targets[:] = self.cur_targets
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.nut_dof_pos_prev[:] = self.nut_dof_pos
        self.ft_rot_prev[:] = self.fingertip_orientation
        self.ft_pos_prev[:] = self.fingertip_pos
        self.dof_vel_prev[:] = self.dof_vel_finite_diff

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['point_cloud_info'] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict['rot_axis_buf'] = self.rot_axis_buf.to(self.rl_device)
        return self.obs_dict

    def step(self, actions, extrin_record: Optional[torch.Tensor] = None):
        # Save extrinsics if evaluating on just one object.
        action_mask = torch.ones_like(actions)
        if self.config['env']['initPose'] == 'screwdriver_inclined':
            action_mask[:, 5:7] = 0.0
        else:
            action_mask[:, 5:9] = 0.0  # mask out pinky, and ring finger actions
        actions = actions * action_mask
        actions = F.pad(actions, (0, 1), value=0.0) # pad the last dim with 0.0 for the nut joint

        if extrin_record is not None and self.config['env']['object']['evalObjectType'] is not None:
            # Put a (z vectors, is done) tuple into the log.
            self.extrin_log.append(
                (extrin_record.detach().cpu().numpy().copy(), self.eval_done_buf.detach().cpu().numpy().copy())
            )

        self.pre_state = self.xhand_hand_dof_pos[0]
        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        # stage 2 buffer
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['point_cloud_info'] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict['rot_axis_buf'] = self.rot_axis_buf.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def capture_frame(self) -> np.ndarray:
        assert self.enable_camera_sensors  # camera sensors should be enabled
        assert self.vid_record_tensor is not None
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        frame = self.vid_record_tensor.cpu().numpy()
        self.gym.end_access_image_tensors(self.sim)

        return frame

    def update_low_level_control(self, step_id):
        previous_dof_pos = self.xhand_hand_dof_pos.clone()
        self._refresh_gym()
        random_action_noise_t = torch.normal(0, self.random_action_noise_t_scale, size=self.xhand_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noise_action = self.cur_targets[..., :self.num_xhand_hand_dofs] + self.random_action_noise_e + random_action_noise_t

        if self.torque_control:
            dof_pos = self.xhand_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff[:, step_id] = dof_vel.clone()
            torques = self.p_gain * (noise_action - dof_pos) - self.d_gain * dof_vel
            torques = torch.clip(torques, -self.torque_limit, self.torque_limit).clone()
            self.torques[:, step_id, :self.num_xhand_hand_dofs] = torques.clone()
            torques = self.torques[:, step_id, :].clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(noise_action))

    def update_rigid_body_force(self):
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for env in self.envs]
            obj_mass = to_torch(obj_mass, device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def check_termination(self, object_pos):
        term_by_max_eps = torch.greater_equal(self.progress_buf, self.max_episode_length)
        resets = term_by_max_eps

        # finger nut distance check
        nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        nut_pos = nut_states[..., :3] 

        thumb_pos = self.rigid_body_states[:, self.fingertip_handles[-1], :3]
        index_pos = self.rigid_body_states[:, self.fingertip_handles[0], :3]
        thumb_dist = torch.norm(thumb_pos - nut_pos, dim=-1)
        index_dist = torch.norm(index_pos - nut_pos, dim=-1)
        # Reset logging
        finger_dist_reset = torch.logical_or(thumb_dist > self.reset_dist_threshold, index_dist > self.reset_dist_threshold)
        resets = torch.logical_or(resets, finger_dist_reset)  

        # nut constant pos check - terminate if nut position is similar over 10 timesteps
        nut_pos_history_filled = (self.progress_buf >= self.nut_termination_history_len)
        nut_pos_variance = torch.var(self.nut_dof_pos_history, dim=1).squeeze(-1)  # (num_envs,)
        nut_pos_stagnant = (nut_pos_variance < self.nut_stagnation_eps) & nut_pos_history_filled
        resets = torch.logical_or(resets, nut_pos_stagnant) 

        # nut contact check - terminate if nut has 0 contact force over 10 timesteps
        contact_history_filled = (self.progress_buf >= self.nut_termination_history_len)
        no_contact = torch.all(self.nut_contact_history <= 1e-3, dim=1).squeeze(-1) 
        no_contact_reset = no_contact & contact_history_filled
        resets = torch.logical_or(resets, no_contact_reset) 

        # screw joint limit check for automatic reset
        if hasattr(self, 'dof_state'):
            current_screw_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_xhand_hand_dofs:]
            current_screw_pos = current_screw_dof_state[:, 0, 0]
            screw_upper_limit = 628.3185
            reset_threshold = screw_upper_limit - 5.0  # Reset when within 5 radians of limit
            
            screw_at_limit = current_screw_pos > reset_threshold
            resets = torch.logical_or(resets, screw_at_limit)
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        self.fingertip_states = self.rigid_body_states[:, self.fingertip_handles]
        self.fingertip_pos = self.fingertip_states[:, :, :3].reshape(self.num_envs, -1)
        self.fingertip_orientation = self.fingertip_states[:, :, 3:7].reshape(self.num_envs, -1)
        self.fingertip_linvel = self.fingertip_states[:, :, 7:10].reshape(self.num_envs, -1)
        self.fingertip_angvel = self.fingertip_states[:, :, 10:13].reshape(self.num_envs, -1)
        self.nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        self.nut_pos = self.nut_states[:, :3]
        all_contact_forces = torch.norm(self.contact_forces.clone(), dim=-1)
        self.nut_contact = all_contact_forces[:, self.screw_nut_rb_handle].unsqueeze(-1)
        self.nut_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_xhand_hand_dofs:]
        self.nut_dof_vel = self.nut_dof_state[:, 0, 1].unsqueeze(-1)
        self.nut_dof_pos = self.nut_dof_state[:, 0, 0].unsqueeze(-1)


    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_screw_joint_friction = rand_config.get('randomizeScrewJointFriction', False)
        self.randomize_screw_joint_friction_lower = rand_config.get('randomizeScrewJointFrictionLower', 0.2)
        self.randomize_screw_joint_friction_upper = rand_config.get('randomizeScrewJointFrictionUpper', 0.2)
        self.randomize_scale = rand_config['randomizeScale']
        self.randomize_hand_scale = rand_config['randomize_hand_scale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        # Store joint-specific randomization parameters
        dof_names = ['left_hand_index_bend_joint', 'left_hand_index_joint1', 'left_hand_index_joint2', 'left_hand_mid_joint1', 'left_hand_mid_joint2', 'left_hand_pinky_joint1', 'left_hand_pinky_joint2', 'left_hand_ring_joint1', 'left_hand_ring_joint2', 'left_hand_thumb_bend_joint', 'left_hand_thumb_rota_joint1', 'left_hand_thumb_rota_joint2']
        self.joint_p_gain_lower = {}
        self.joint_p_gain_upper = {}
        self.joint_d_gain_lower = {}
        self.joint_d_gain_upper = {}
        for joint_name in dof_names:
            p_lower_key = f'randomizePGainLower_{joint_name}'
            p_upper_key = f'randomizePGainUpper_{joint_name}'
            d_lower_key = f'randomizeDGainLower_{joint_name}'
            d_upper_key = f'randomizeDGainUpper_{joint_name}'
            self.joint_p_gain_lower[joint_name] = rand_config.get(p_lower_key, rand_config['randomizePGainLower'])
            self.joint_p_gain_upper[joint_name] = rand_config.get(p_upper_key, rand_config['randomizePGainUpper'])
            self.joint_d_gain_lower[joint_name] = rand_config.get(d_lower_key, rand_config['randomizeDGainLower'])
            self.joint_d_gain_upper[joint_name] = rand_config.get(d_upper_key, rand_config['randomizeDGainUpper'])

        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.random_obs_noise_e_scale = rand_config['obs_noise_e_scale']
        self.random_obs_noise_t_scale = rand_config['obs_noise_t_scale']
        self.random_pose_noise = rand_config['pose_noise_scale']
        self.random_action_noise_e_scale = rand_config['action_noise_e_scale']
        self.random_action_noise_t_scale = rand_config['action_noise_t_scale']
        # stage 2 specific
        self.noisy_rpy_scale = rand_config['noisy_rpy_scale']
        self.noisy_pos_scale = rand_config['noisy_pos_scale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']
        self.contact_input_dim = p_config['contact_input_dim']
        self.contact_form = p_config['contact_form']
        self.contact_input = p_config['contact_input']
        self.contact_binarize_threshold = p_config['contact_binarize_threshold']
        self.enable_priv_obj_orientation = p_config['enable_obj_orientation']
        self.enable_priv_obj_linvel = p_config['enable_obj_linvel']
        self.enable_priv_obj_angvel = p_config['enable_obj_angvel']
        self.enable_priv_fingertip_position = p_config['enable_ft_pos']
        self.enable_priv_fingertip_orientation = p_config['enable_ft_orientation']
        self.enable_priv_fingertip_linvel = p_config['enable_ft_linvel']
        self.enable_priv_fingertip_angvel = p_config['enable_ft_angvel']
        self.enable_priv_hand_scale = p_config['enable_hand_scale']
        self.enable_priv_obj_restitution = p_config['enable_obj_restitution']
        self.enable_priv_tactile = p_config['enable_tactile']
        self.enable_priv_nut_contact = p_config['enable_nut_contact']
        self.enable_priv_nut_pos = p_config['enable_nut_pos']
        self.enable_priv_nut_dof_vel = p_config['enable_nut_dof_vel']
        self.enable_priv_nut_dof_pos = p_config['enable_nut_dof_pos']
        self.enable_priv_hand_position = p_config['enable_hand_position']
        self.enable_priv_hand_orientation = p_config['enable_hand_orientation']
        self.enable_priv_hand_joint_pos = p_config['enable_hand_joint_pos']
        self.enable_priv_pgain = p_config['enable_pgain']
        self.enable_priv_dgain = p_config['enable_dgain']
        self.enable_priv_screw_joint_friction = p_config.get('enable_screw_joint_friction', False)
        self.num_contacts = 0

        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }

        start_index = 0
        priv_dims = OrderedDict()
        priv_dims['obj_orientation'] = 4
        priv_dims['obj_linvel'] = 3
        priv_dims['obj_angvel'] = 3
        priv_dims['fingertip_position'] = 3 * self.fingers_num
        priv_dims['fingertip_orientation'] = 4 * self.fingers_num
        priv_dims['fingertip_linvel'] = self.fingers_num * 3
        priv_dims['fingertip_angvel'] = self.fingers_num * 3
        priv_dims['hand_scale'] = 1
        priv_dims['obj_restitution'] = 1
        priv_dims['tactile'] = self.num_contacts
        priv_dims['nut_contact'] = 1
        priv_dims['nut_pos'] = 3
        priv_dims['nut_dof_vel'] = 1
        priv_dims['nut_dof_pos'] = 1
        priv_dims['pgain'] = self.numActions
        priv_dims['dgain'] = self.numActions
        priv_dims['hand_joint_pos'] = self.numActions
        priv_dims['hand_orientation'] = 4
        priv_dims['hand_position'] = 3
        priv_dims['screw_joint_friction'] = 1
        for name, dim in priv_dims.items():
            if eval(f'self.enable_priv_{name}'):
                self.priv_info_dict[name] = (start_index, start_index + dim)
                start_index += dim

    def _update_priv_buf(self, env_id, name, value):
        # normalize to -1, 1
        if eval(f'self.enable_priv_{name}'):
            s, e = self.priv_info_dict[name]
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            self.priv_info_buf[env_id, s:e] = value

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)
        
        # Load object tilt configuration (default to True for backward compatibility)
        self.object_tilt_enabled = o_config.get('object_tilt', True)
        print(f'---- Object Tilt Configuration ----')
        print(f'Object tilt enabled: {self.object_tilt_enabled}')

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ball.urdf',
            'simple_cube': 'assets/cube.urdf',
            'simple_cylin4cube': 'assets/cylinder4cube.urdf',
        }
        for p_id, prim in enumerate(primitive_list):
            if 'screw' in prim:
                subset_name = self.object_type.split('_')[-1] if '_' in self.object_type else 'None'
                screw_compounds = sorted(glob(f'assets/screw/{subset_name}/*.urdf'))
                screw_compound_list = [f'screw_{i}' for i in range(len(screw_compounds))]
                self.object_type_list += screw_compound_list
                for i, name in enumerate(screw_compounds):
                    self.asset_files_dict[f'screw_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(screw_compound_list) for _ in screw_compound_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(f'using {len(self.object_type_list)} training objects')
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.priv_info_dim = max([v[1] for k, v in self.priv_info_dict.items()])
        self.point_cloud_sampled_dim = self.config['env']['hora']['point_cloud_sampled_dim']
        self.point_cloud_buffer_dim = self.point_cloud_sampled_dim
        self.priv_info_buf = torch.zeros((num_envs, self.priv_info_dim), device=self.device, dtype=torch.float)
        # for collecting bc data
        self.point_cloud_buf = torch.zeros((num_envs, self.point_cloud_sampled_dim, 3), device=self.device, dtype=torch.float)
        # fixed noise per-episode, for different hardware have different this value
        self.random_obs_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        self.random_action_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        # ---- stage 2 buffers
        # stage 2 related buffers
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, self.numActions*2), device=self.device, dtype=torch.float)
        
        # ---- nut termination buffers (10 timesteps history)
        self.nut_dof_pos_history = torch.zeros((num_envs, self.nut_termination_history_len, 1), device=self.device, dtype=torch.float)
        self.nut_contact_history = torch.zeros((num_envs, self.nut_termination_history_len, 1), device=self.device, dtype=torch.float)
        self.last_nut_pos = torch.zeros((num_envs,), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        # the list
        self.reward_scale_dict = {}
        for k, v in r_config.items():
            if 'scale' in k:
                if type(v) is not omegaconf.listconfig.ListConfig:
                    v = [v, v, 0, 0]
                else:
                    assert len(v) == 4
                self.reward_scale_dict[k.replace('_scale', '')] = v
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        
        # angular velocity penalty threshold curriculum
        angvel_threshold_config = r_config.get('angvelPenaltyThres', 10.0)
        if type(angvel_threshold_config) is not omegaconf.listconfig.ListConfig:
            # No curriculum - use fixed value
            self.angvel_penalty_threshold_curriculum = [angvel_threshold_config, angvel_threshold_config, 0, 0]
        else:
            # Curriculum enabled - [init_threshold, final_threshold, start_step, end_step]
            assert len(angvel_threshold_config) == 4
            self.angvel_penalty_threshold_curriculum = angvel_threshold_config
        
        # Initialize with the starting threshold
        self.angvel_penalty_threshold = self.angvel_penalty_threshold_curriculum[0]

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset'] # load hand asset
        
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.convex_decomposition_from_submeshes = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        else:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in
                                ['left_hand_index_rota_tip', 
                                'left_hand_mid_tip',  'left_hand_pinky_tip', 'left_hand_ring_tip', 'left_hand_thumb_rota_tip']]

        # load object asset
        self.object_asset_list = []
        self.asset_point_clouds = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.fix_base_link = True
            # If we've specified a specific eval object, we only need to load that object.
            eval_object_type = self.config['env']['object']['evalObjectType']
            if eval_object_type is not None and object_type != eval_object_type:
                self.object_asset_list.append(None)
                self.asset_point_clouds.append(None)
                continue

            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)
            if 'screw' in object_type and self.point_cloud_sampled_dim > 0:
                # Load pre-generated point cloud from NPY file (following cylinder/cube pattern)
                pc_file = os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy'))
                if os.path.exists(pc_file):
                    screw_points = np.load(pc_file)
                    
                    # Resample to desired number of points if needed
                    if len(screw_points) != self.point_cloud_sampled_dim:
                        if len(screw_points) > self.point_cloud_sampled_dim:
                            indices = np.random.choice(len(screw_points), self.point_cloud_sampled_dim, replace=False)
                        else:
                            indices = np.random.choice(len(screw_points), self.point_cloud_sampled_dim, replace=True)
                        screw_points = screw_points[indices]
                    
                    self.asset_point_clouds.append(screw_points)
                else:
                    # Fallback: simple cylinder if NPY file missing (approximate nut size)
                    fallback_points = sample_cylinder(1.0) * 0.022  # Match typical nut radius
                    self.asset_point_clouds.append(fallback_points)
            else:
                # Default fallback for unknown object types
                if self.point_cloud_sampled_dim > 0:
                    fallback_points = sample_cylinder(1.0) * 0.05
                    self.asset_point_clouds.append(fallback_points)

        assert any([x is not None for x in self.object_asset_list])

    def _parse_hand_dof_props(self):
        self.num_xhand_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        xhand_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.xhand_hand_dof_lower_limits = []
        self.xhand_hand_dof_upper_limits = []
        
        hand_asset_file = self.config['env']['asset']['handAsset']
        xhand_dof_lower_limits = [-0.175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.05, -0.17,]
        self.xhand_dof_lower_limits = np.array(xhand_dof_lower_limits)

        xhand_dof_upper_limits = [0.175, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.83, 1.57, 1.83,]
        if self.config['env']['object']['thumb_range_limit']:
            xhand_dof_upper_limits[9] = 1.73
            xhand_dof_lower_limits[9] = 0.6
        self.xhand_dof_upper_limits = np.array(xhand_dof_upper_limits)

        # Effort limits from the actuatorfrcrange in the MJCF file
        xhand_effort_limits = [0.4, 1.1, 0.4, 1.1, 0.4, 1.1, 0.4, 1.1, 1.1, 0.4, 1.1, 1.1, 0.4,]
        
        for i in range(self.num_xhand_hand_dofs):
            # Set the joint limits based on the URDF
            xhand_hand_dof_props['lower'][i] = xhand_dof_lower_limits[i]
            xhand_hand_dof_props['upper'][i] = xhand_dof_upper_limits[i]
            self.xhand_hand_dof_lower_limits.append(xhand_dof_lower_limits[i])
            self.xhand_hand_dof_upper_limits.append(xhand_dof_upper_limits[i])
            
            # Set the effort limit
            xhand_hand_dof_props['effort'][i] = xhand_effort_limits[i]
            
            # Set controller properties
            if self.torque_control:
                xhand_hand_dof_props['stiffness'][i] = 0.
                xhand_hand_dof_props['damping'][i] = 0.
                xhand_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                xhand_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                xhand_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            
            # From URDF dynamics, all joints have damping=1 and friction=1
            xhand_hand_dof_props['friction'][i] = 0.01
            xhand_hand_dof_props['armature'][i] = 0.001

        self.xhand_hand_dof_lower_limits = to_torch(self.xhand_hand_dof_lower_limits, device=self.device)
        self.xhand_hand_dof_upper_limits = to_torch(self.xhand_hand_dof_upper_limits, device=self.device)
        return xhand_hand_dof_props

    def _init_object_pose(self):
        hand_asset_file = self.config['env']['asset']['handAsset']
        
        xhand_hand_start_pose = gymapi.Transform()
        xhand_hand_start_pose.p = gymapi.Vec3(0, 0, 0.21)
        xhand_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi/2 - 25 * (np.pi / 180))
                                    
        # Object position relative to hand
        pose_dx, pose_dy, pose_dz = 0.00, 0.00, 0.00
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = xhand_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = xhand_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = xhand_hand_start_pose.p.z + pose_dz

        return xhand_hand_start_pose, object_start_pose

    @staticmethod
    def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., 3:])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        return quaternions[..., :3] / sin_half_angles_over_angles

def compute_hand_reward(
    rotate_reward, rotate_reward_scale: float,
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
    z_dist_penalty, z_dist_penalty_scale: float,
    rotate_penalty, rotate_penalty_scale: float,
    proximity_reward, proximity_reward_scale: float
):
    reward = rotate_reward_scale * rotate_reward
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    reward = reward + z_dist_penalty * z_dist_penalty_scale
    reward = reward + rotate_penalty * rotate_penalty_scale
    reward = reward + proximity_reward * proximity_reward_scale
    return reward
