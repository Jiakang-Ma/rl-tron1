"""
模仿学习奖励函数
Imitation Learning Reward Functions

用于跟踪参考动作轨迹的奖励函数集合。
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
    动作模仿奖励类 / Motion Imitation Reward Class
    
    跟踪参考动作轨迹，计算关节位置和基座高度的跟踪奖励。
    Tracks reference motion trajectory, computes tracking rewards for joint positions and base height.
    
    使用方法 / Usage:
        在 RewardsCfg 中配置:
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
        初始化模仿奖励 / Initialize imitation reward.
        
        Args:
            cfg: 奖励配置 / Reward configuration
            env: RL 环境 / RL environment
        """
        super().__init__(cfg, env)
        
        # 获取参数
        self.motion_file = cfg.params.get("motion_file", "motion_data/motions/jump.yaml")
        self.joint_pos_scale = cfg.params.get("joint_pos_scale", 5.0)
        self.base_height_scale = cfg.params.get("base_height_scale", 10.0)
        self.joint_vel_scale = cfg.params.get("joint_vel_scale", 0.1)
        
        # 关节名称顺序 (与 motion_loader 一致)
        self.joint_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
        ]
        
        # 加载参考动作
        self._load_motion()
        
        # 每个环境的动作相位 [0, duration)
        self.motion_phase = torch.zeros(env.num_envs, device=self._env.device)
        
        # 获取机器人资产
        self.robot: Articulation = env.scene["robot"]
        
        # 获取关节索引
        self.joint_indices = []
        for name in self.joint_names:
            idx = self.robot.find_joints(name)[0]
            if len(idx) > 0:
                self.joint_indices.append(idx[0])
        self.joint_indices = torch.tensor(self.joint_indices, device=self._env.device)
        
    def _load_motion(self):
        """加载参考动作数据 / Load reference motion data"""
        import sys
        import yaml
        from scipy.interpolate import CubicSpline
        
        # 构建绝对路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
        motion_path = os.path.join(project_root, self.motion_file)
        
        if not os.path.exists(motion_path):
            print(f"[WARNING] Motion file not found: {motion_path}")
            print("[WARNING] Using dummy motion data")
            # 使用虚拟数据
            self.duration = 1.0
            self._ref_joint_pos = lambda t: np.zeros(6)
            self._ref_joint_vel = lambda t: np.zeros(6)
            self._ref_base_height = lambda t: 0.55
            return
            
        # 加载 YAML
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
        
        # 创建插值器
        self._joint_splines = []
        for i in range(6):
            spline = CubicSpline(times, joint_positions[:, i], bc_type='periodic')
            self._joint_splines.append(spline)
            
        self._height_spline = CubicSpline(times, base_heights, bc_type='periodic')
        
        # 定义查询函数
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
        计算模仿奖励 / Compute imitation reward
        
        Returns:
            torch.Tensor: 每个环境的奖励值
        """
        # 更新相位 (按仿真时间步进)
        dt = env.step_dt
        self.motion_phase = (self.motion_phase + dt) % self.duration
        
        # 获取当前状态
        current_joint_pos = self.robot.data.joint_pos[:, self.joint_indices]
        current_base_height = self.robot.data.root_pos_w[:, 2]
        
        # 计算每个环境的参考值
        num_envs = env.num_envs
        ref_joint_pos_batch = torch.zeros(num_envs, 6, device=self._env.device)
        ref_height_batch = torch.zeros(num_envs, device=self._env.device)
        
        for i in range(num_envs):
            t = self.motion_phase[i].item()
            ref_joint_pos_batch[i] = torch.tensor(self._ref_joint_pos(t), device=self._env.device)
            ref_height_batch[i] = self._ref_base_height(t)
            
        # 计算关节位置误差奖励
        joint_pos_error = current_joint_pos - ref_joint_pos_batch
        joint_pos_reward = torch.exp(-self.joint_pos_scale * torch.sum(joint_pos_error ** 2, dim=-1))
        
        # 计算基座高度误差奖励
        height_error = current_base_height - ref_height_batch
        height_reward = torch.exp(-self.base_height_scale * (height_error ** 2))
        
        # 组合奖励 (可调整权重)
        total_reward = 0.7 * joint_pos_reward + 0.3 * height_reward
        
        return total_reward
        
    def reset(self, env_ids: torch.Tensor):
        """重置指定环境的相位 / Reset phase for specified environments"""
        self.motion_phase[env_ids] = 0.0


def jump_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    跳跃高度奖励 / Jump height reward
    
    奖励机器人达到目标跳跃高度
    Rewards the robot for reaching target jump height
    
    改进: 使用指数奖励，站立高度 0.82m 作为基准
    Improved: Uses exponential reward with standing height 0.82m as baseline
    
    Args:
        env: RL 环境
        target_height: 目标跳跃高度 (m)
        asset_cfg: 机器人配置
        
    Returns:
        torch.Tensor: 高度奖励
    """
    robot: Articulation = env.scene[asset_cfg.name]
    current_height = robot.data.root_pos_w[:, 2]
    
    # 站立高度基准
    standing_height = 0.82
    
    # 计算超过站立高度的部分
    height_above_standing = current_height - standing_height
    
    # 只有高度超过站立高度才给奖励
    # 使用 softplus 来平滑处理
    height_bonus = torch.clamp(height_above_standing, min=0.0)
    
    # 指数奖励: 越高奖励越大
    reward = torch.exp(2.0 * height_bonus) - 1.0
    
    # 额外奖励: 如果达到目标高度
    reached_target = current_height > target_height
    reward = reward + 2.0 * reached_target.float()
    
    return reward


def air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_air_time: float = 0.3,
) -> torch.Tensor:
    """
    滞空时间奖励 / Air time reward
    
    奖励机器人在空中停留的时间
    Rewards the robot for time spent in the air
    
    Args:
        env: RL 环境
        sensor_cfg: 接触传感器配置
        target_air_time: 目标滞空时间 (s)
        
    Returns:
        torch.Tensor: 滞空奖励
    """
    from isaaclab.sensors import ContactSensor
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取双脚的空中时间
    air_time = contact_sensor.data.current_air_time
    
    # 计算两脚同时在空中的奖励
    both_feet_air = (air_time[:, 0] > 0.05) & (air_time[:, 1] > 0.05)
    
    # 奖励滞空时间
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
    落地稳定性奖励 / Landing stability reward
    
    惩罚落地时的不稳定姿态
    Penalizes unstable poses during landing
    
    Args:
        env: RL 环境
        asset_cfg: 机器人配置
        
    Returns:
        torch.Tensor: 稳定性奖励
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 获取角速度
    ang_vel = robot.data.root_ang_vel_w
    
    # 惩罚大的角速度 (不稳定)
    ang_vel_magnitude = torch.norm(ang_vel, dim=-1)
    stability_reward = torch.exp(-0.5 * ang_vel_magnitude)
    
    return stability_reward
