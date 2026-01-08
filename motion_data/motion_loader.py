"""
Reference motion loader
Loads motion keyframes from YAML files and generates smooth trajectories
"""

import os
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from scipy.interpolate import CubicSpline


@dataclass
class MotionData:
    """Single frame motion data"""
    time: float
    joint_positions: np.ndarray  # 6 joints: [abad_L, hip_L, knee_L, abad_R, hip_R, knee_R]
    base_height: float
    base_velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    base_orientation: Optional[np.ndarray] = None  # [roll, pitch, yaw]


class MotionLoader:
    """
    Reference motion loader
    
    Usage:
        loader = MotionLoader('motion_data/motions/jump.yaml')
        state = loader.get_state(0.5)  # Get state at t=0.5s
        state = loader.get_state_by_phase(0.25)  # Get state at 25% phase
    """
    
    # TRON1 joint name order
    JOINT_NAMES = [
        "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
        "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
    ]
    
    # Joint angle limits (rad)
    JOINT_LIMITS = {
        "abad_L_Joint": (-0.38, 1.40),
        "hip_L_Joint": (-1.01, 1.40),
        "knee_L_Joint": (-0.87, 1.36),
        "abad_R_Joint": (-1.40, 0.38),
        "hip_R_Joint": (-1.40, 1.01),
        "knee_R_Joint": (-1.36, 0.87),
    }
    
    def __init__(self, motion_file: str):
        """
        Initialize loader
        
        Args:
            motion_file: YAML motion file path
        """
        self.motion_file = motion_file
        self.keyframes: List[MotionData] = []
        self.duration = 0.0
        self.loop = True
        
        # Interpolators
        self._joint_splines: Optional[List[CubicSpline]] = None
        self._height_spline: Optional[CubicSpline] = None
        
        self._load_motion()
        self._build_splines()
        
    def _load_motion(self):
        """Load motion data from YAML"""
        with open(self.motion_file, 'r') as f:
            data = yaml.safe_load(f)
            
        motion_info = data.get('motion', {})
        self.name = motion_info.get('name', 'unnamed')
        self.duration = motion_info.get('duration', 1.0)
        self.loop = motion_info.get('loop', True)
        
        keyframes_data = data.get('keyframes', [])
        
        for kf in keyframes_data:
            time = kf['time']
            joints = kf['joints']
            
            # Extract joint angles in order
            joint_positions = np.array([
                joints.get('abad_L', 0.0),
                joints.get('hip_L', 0.0),
                joints.get('knee_L', 0.0),
                joints.get('abad_R', 0.0),
                joints.get('hip_R', 0.0),
                joints.get('knee_R', 0.0),
            ])
            
            base_height = kf.get('base_height', 0.55)
            
            # Optional base velocity and orientation
            base_vel = kf.get('base_velocity')
            if base_vel:
                base_vel = np.array([base_vel.get('x', 0), base_vel.get('y', 0), base_vel.get('z', 0)])
                
            base_orient = kf.get('base_orientation')
            if base_orient:
                base_orient = np.array([base_orient.get('roll', 0), base_orient.get('pitch', 0), base_orient.get('yaw', 0)])
            
            self.keyframes.append(MotionData(
                time=time,
                joint_positions=joint_positions,
                base_height=base_height,
                base_velocity=base_vel,
                base_orientation=base_orient,
            ))
            
        # Ensure sorted by time
        self.keyframes.sort(key=lambda x: x.time)
        
        print(f"Loaded motion '{self.name}': {len(self.keyframes)} keyframes, duration={self.duration}s")
        
    def _build_splines(self):
        """Build cubic spline interpolators"""
        if len(self.keyframes) < 2:
            raise ValueError("Need at least 2 keyframes for interpolation")
            
        times = np.array([kf.time for kf in self.keyframes])
        
        # Joint position interpolators (one per joint)
        joint_positions = np.array([kf.joint_positions for kf in self.keyframes])
        self._joint_splines = []
        for i in range(6):
            spline = CubicSpline(times, joint_positions[:, i], bc_type='periodic' if self.loop else 'natural')
            self._joint_splines.append(spline)
            
        # Base height interpolator
        heights = np.array([kf.base_height for kf in self.keyframes])
        self._height_spline = CubicSpline(times, heights, bc_type='periodic' if self.loop else 'natural')
        
    def get_state(self, time: float) -> MotionData:
        """
        Get motion state at specified time
        
        Args:
            time: Time (seconds)
            
        Returns:
            MotionData: Interpolated motion state
        """
        # Handle time looping
        if self.loop:
            time = time % self.duration
        else:
            time = np.clip(time, 0, self.duration)
            
        # Interpolate joint positions
        joint_positions = np.array([spline(time) for spline in self._joint_splines])
        
        # Clip to joint limits
        for i, name in enumerate(self.JOINT_NAMES):
            low, high = self.JOINT_LIMITS[name]
            joint_positions[i] = np.clip(joint_positions[i], low, high)
            
        # Interpolate base height
        base_height = float(self._height_spline(time))
        
        return MotionData(
            time=time,
            joint_positions=joint_positions,
            base_height=base_height,
        )
        
    def get_state_by_phase(self, phase: float) -> MotionData:
        """
        Get motion state by phase
        
        Args:
            phase: Phase [0, 1]
            
        Returns:
            MotionData: Interpolated motion state
        """
        time = phase * self.duration
        return self.get_state(time)
        
    def get_joint_velocities(self, time: float) -> np.ndarray:
        """
        Get joint velocities (interpolation derivative)
        
        Args:
            time: Time (seconds)
            
        Returns:
            np.ndarray: 6 joint velocities
        """
        if self.loop:
            time = time % self.duration
        else:
            time = np.clip(time, 0, self.duration)
            
        velocities = np.array([spline(time, 1) for spline in self._joint_splines])
        return velocities
        
    def validate_motion(self) -> bool:
        """
        Validate motion is within joint limits
        
        Returns:
            bool: Whether valid
        """
        valid = True
        for kf in self.keyframes:
            for i, name in enumerate(self.JOINT_NAMES):
                low, high = self.JOINT_LIMITS[name]
                val = kf.joint_positions[i]
                if val < low or val > high:
                    print(f"Warning: {name} = {val:.3f} out of range [{low:.3f}, {high:.3f}] at t={kf.time:.2f}s")
                    valid = False
        return valid


if __name__ == '__main__':
    # Test loader
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    motion_file = os.path.join(script_dir, 'motions', 'jump.yaml')
    
    if os.path.exists(motion_file):
        loader = MotionLoader(motion_file)
        loader.validate_motion()
        
        # Test interpolation
        for t in np.linspace(0, loader.duration, 11):
            state = loader.get_state(t)
            print(f"t={t:.2f}s: height={state.base_height:.3f}m, joints={state.joint_positions}")
    else:
        print(f"Motion file not found: {motion_file}")
