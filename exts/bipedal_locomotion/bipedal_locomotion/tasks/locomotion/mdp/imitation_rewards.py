"""
Imitation Learning Reward Functions

A collection of reward functions for tracking reference motion trajectories.
"""

from __future__ import annotations

import os
import torch
import numpy as np
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


class MotionImitationReward(ManagerTermBase):
    """
    Motion Imitation Reward Class
    
    Tracks reference motion trajectory, computes tracking rewards for joint positions and base height.
    
    Usage:
        Configure in RewardsCfg:
        imitation_reward = RewTerm(
            func=mdp.MotionImitationReward,
            weight=5.0,
            params={
                "motion_file": "motion_data/motions/jump.yaml",
                "joint_pos_scale": 5.0,
                "base_height_scale": 10.0,
            }
        )
    """
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """
        Initialize imitation reward.
        
        Args:
            cfg: Reward configuration
            env: RL environment
        """
        super().__init__(cfg, env)
        
        # Get parameters
        self.motion_file = cfg.params.get("motion_file", "motion_data/motions/jump.yaml")
        self.joint_pos_scale = cfg.params.get("joint_pos_scale", 5.0)
        self.base_height_scale = cfg.params.get("base_height_scale", 10.0)
        self.joint_vel_scale = cfg.params.get("joint_vel_scale", 0.1)
        
        # Joint name order (consistent with motion_loader)
        self.joint_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
        ]
        
        # Load reference motion
        self._load_motion()
        
        # Motion phase for each environment [0, duration)
        self.motion_phase = torch.zeros(env.num_envs, device=self._env.device)
        
        # Get robot asset
        self.robot: Articulation = env.scene["robot"]
        
        # Get joint indices
        self.joint_indices = []
        for name in self.joint_names:
            idx = self.robot.find_joints(name)[0]
            if len(idx) > 0:
                self.joint_indices.append(idx[0])
        self.joint_indices = torch.tensor(self.joint_indices, device=self._env.device)
        
    def _load_motion(self):
        """Load reference motion data"""
        import sys
        import yaml
        from scipy.interpolate import CubicSpline
        
        # Build absolute path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
        motion_path = os.path.join(project_root, self.motion_file)
        
        if not os.path.exists(motion_path):
            print(f"[WARNING] Motion file not found: {motion_path}")
            print("[WARNING] Using dummy motion data")
            # Use dummy data
            self.duration = 1.0
            self._ref_joint_pos = lambda t: np.zeros(6)
            self._ref_joint_vel = lambda t: np.zeros(6)
            self._ref_base_height = lambda t: 0.55
            return
            
        # Load YAML
        with open(motion_path, 'r') as f:
            data = yaml.safe_load(f)
            
        motion_info = data.get('motion', {})
        self.duration = motion_info.get('duration', 1.0)
        
        keyframes = data.get('keyframes', [])
        
        times = []
        joint_positions = []
        base_heights = []
        
        for kf in keyframes:
            times.append(kf['time'])
            joints = kf['joints']
            joint_positions.append([
                joints.get('abad_L', 0.0),
                joints.get('hip_L', 0.0),
                joints.get('knee_L', 0.0),
                joints.get('abad_R', 0.0),
                joints.get('hip_R', 0.0),
                joints.get('knee_R', 0.0),
            ])
            base_heights.append(kf.get('base_height', 0.55))
            
        times = np.array(times)
        joint_positions = np.array(joint_positions)
        base_heights = np.array(base_heights)
        
        # Create interpolators
        self._joint_splines = []
        for i in range(6):
            spline = CubicSpline(times, joint_positions[:, i], bc_type='periodic')
            self._joint_splines.append(spline)
            
        self._height_spline = CubicSpline(times, base_heights, bc_type='periodic')
        
        # Define query functions
        def ref_joint_pos(t):
            t = t % self.duration
            return np.array([spline(t) for spline in self._joint_splines])
            
        def ref_joint_vel(t):
            t = t % self.duration
            return np.array([spline(t, 1) for spline in self._joint_splines])
            
        def ref_base_height(t):
            t = t % self.duration
            return float(self._height_spline(t))
            
        self._ref_joint_pos = ref_joint_pos
        self._ref_joint_vel = ref_joint_vel
        self._ref_base_height = ref_base_height
        
        print(f"[INFO] Loaded motion '{motion_info.get('name', 'unnamed')}': duration={self.duration}s")
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        motion_file: str,
        joint_pos_scale: float,
        base_height_scale: float,
    ) -> torch.Tensor:
        """
        Compute imitation reward
        
        Returns:
            torch.Tensor: Reward value for each environment
        """
        # Update phase (step by simulation time)
        dt = env.step_dt
        self.motion_phase = (self.motion_phase + dt) % self.duration
        
        # Get current state
        current_joint_pos = self.robot.data.joint_pos[:, self.joint_indices]
        current_base_height = self.robot.data.root_pos_w[:, 2]
        
        # Calculate reference values for each environment
        num_envs = env.num_envs
        ref_joint_pos_batch = torch.zeros(num_envs, 6, device=self._env.device)
        ref_height_batch = torch.zeros(num_envs, device=self._env.device)
        
        for i in range(num_envs):
            t = self.motion_phase[i].item()
            ref_joint_pos_batch[i] = torch.tensor(self._ref_joint_pos(t), device=self._env.device)
            ref_height_batch[i] = self._ref_base_height(t)
            
        # Calculate joint position error reward
        joint_pos_error = current_joint_pos - ref_joint_pos_batch
        joint_pos_reward = torch.exp(-self.joint_pos_scale * torch.sum(joint_pos_error ** 2, dim=-1))
        
        # Calculate base height error reward
        height_error = current_base_height - ref_height_batch
        height_reward = torch.exp(-self.base_height_scale * (height_error ** 2))
        
        # Combine rewards (weights can be adjusted)
        total_reward = 0.7 * joint_pos_reward + 0.3 * height_reward
        
        return total_reward
        
    def reset(self, env_ids: torch.Tensor):
        """Reset phase for specified environments"""
        self.motion_phase[env_ids] = 0.0


def jump_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Jump height reward
    
    Rewards the robot for reaching target jump height
    
    Improved: Uses exponential reward with standing height 0.82m as baseline
    
    Args:
        env: RL environment
        target_height: Target jump height (m)
        asset_cfg: Robot configuration
        
    Returns:
        torch.Tensor: Height reward
    """
    robot: Articulation = env.scene[asset_cfg.name]
    current_height = robot.data.root_pos_w[:, 2]
    
    # Standing height baseline
    standing_height = 0.82
    
    # Calculate height above standing
    height_above_standing = current_height - standing_height
    
    # Only reward when height exceeds standing height
    # Use softplus for smooth handling
    height_bonus = torch.clamp(height_above_standing, min=0.0)
    
    # Exponential reward: higher = more reward
    reward = torch.exp(2.0 * height_bonus) - 1.0
    
    # Extra reward: if target height is reached
    reached_target = current_height > target_height
    reward = reward + 2.0 * reached_target.float()
    
    return reward


def air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_air_time: float = 0.3,
) -> torch.Tensor:
    """
    Air time reward
    
    Rewards the robot for time spent in the air
    
    Args:
        env: RL environment
        sensor_cfg: Contact sensor configuration
        target_air_time: Target air time (s)
        
    Returns:
        torch.Tensor: Air time reward
    """
    from isaaclab.sensors import ContactSensor
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get air time for both feet
    air_time = contact_sensor.data.current_air_time
    
    # Calculate reward for both feet being in the air
    both_feet_air = (air_time[:, 0] > 0.05) & (air_time[:, 1] > 0.05)
    
    # Reward air time
    avg_air_time = air_time.mean(dim=-1)
    reward = torch.where(
        both_feet_air,
        torch.clamp(avg_air_time / target_air_time, max=1.0),
        torch.zeros_like(avg_air_time)
    )
    
    return reward


def landing_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Landing stability reward
    
    Penalizes unstable poses during landing
    
    Args:
        env: RL environment
        asset_cfg: Robot configuration
        
    Returns:
        torch.Tensor: Stability reward
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get angular velocity
    ang_vel = robot.data.root_ang_vel_w
    
    # Penalize large angular velocities (unstable)
    ang_vel_magnitude = torch.norm(ang_vel, dim=-1)
    stability_reward = torch.exp(-0.5 * ang_vel_magnitude)
    
    return stability_reward
