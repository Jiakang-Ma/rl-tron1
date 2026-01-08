#!/usr/bin/env python3
"""
Reference motion visualization tool
Plays reference motions in MuJoCo to verify physical feasibility

Usage:
    /Users/majk/Library/Python/3.9/bin/mjpython motion_data/visualize_motion.py
    /Users/majk/Library/Python/3.9/bin/mjpython motion_data/visualize_motion.py --motion hop
    /Users/majk/Library/Python/3.9/bin/mjpython motion_data/visualize_motion.py --speed 0.5

Controls:
    - SPACE: Pause/Resume
    - R: Reset to start
    - +/-: Speed up/slow down
    - ESC: Exit
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer as viewer

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from motion_data.motion_loader import MotionLoader


class MotionVisualizer:
    """Reference motion visualizer"""
    
    def __init__(self, motion_file: str, robot_xml: str, speed: float = 1.0):
        """
        Initialize visualizer
        
        Args:
            motion_file: Motion YAML file path
            robot_xml: Robot XML file path
            speed: Playback speed multiplier
        """
        # Load motion
        self.motion = MotionLoader(motion_file)
        print(f"\nLoaded motion: {self.motion.name}")
        print(f"Duration: {self.motion.duration}s")
        print(f"Loop: {self.motion.loop}")
        
        # Validate motion
        if not self.motion.validate_motion():
            print("\n⚠️  Warning: Motion has frames exceeding joint limits!")
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(robot_xml)
        self.data = mujoco.MjData(self.model)
        
        # Playback control
        self.speed = speed
        self.paused = False
        self.motion_time = 0.0
        
        # Get joint indices
        self._get_joint_indices()
        
    def _get_joint_indices(self):
        """Get joint indices in qpos"""
        self.joint_indices = []
        joint_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
        ]
        for name in joint_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx >= 0:
                # qpos index (skip base 7 DOF)
                qpos_idx = self.model.jnt_qposadr[idx]
                self.joint_indices.append(qpos_idx)
            else:
                print(f"Warning: Joint {name} not found")
                
    def key_callback(self, keycode):
        """Keyboard callback"""
        if keycode == ord(' '):
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")
        elif keycode == ord('R') or keycode == ord('r'):
            self.motion_time = 0.0
            print("Reset")
        elif keycode == ord('=') or keycode == ord('+'):
            self.speed = min(self.speed * 1.5, 5.0)
            print(f"Speed: {self.speed:.2f}x")
        elif keycode == ord('-'):
            self.speed = max(self.speed / 1.5, 0.1)
            print(f"Speed: {self.speed:.2f}x")
            
    def set_robot_pose(self, state):
        """Set robot pose (automatically calculate correct base height)"""
        # First reset
        mujoco.mj_resetData(self.model, self.data)
        
        # Set joint angles
        for i, qpos_idx in enumerate(self.joint_indices):
            self.data.qpos[qpos_idx] = state.joint_positions[i]
        
        # Temporarily set high base position to compute FK
        self.data.qpos[0] = 0  # x
        self.data.qpos[1] = 0  # y  
        self.data.qpos[2] = 1.5  # z (temporary high position)
        self.data.qpos[3] = 1  # quat w
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = 0
        
        # Compute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get left and right foot positions
        foot_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_collision")
        foot_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_R_collision")
        
        if foot_l_id >= 0 and foot_r_id >= 0:
            foot_l_pos = self.data.geom_xpos[foot_l_id]
            foot_r_pos = self.data.geom_xpos[foot_r_id]
            
            # Take lowest point of both feet
            foot_z = min(foot_l_pos[2], foot_r_pos[2])
            
            # Foot ball radius
            foot_radius = 0.032
            
            # Calculate required base height adjustment
            # Current foot bottom should be at foot_radius height (just touching ground)
            # Required base height = current base height - (current foot z - target foot z)
            target_foot_z = foot_radius  # Target: foot just touching ground
            
            # If in jump phase (base_height > 0.85), let feet leave ground
            if state.base_height > 0.85:
                # Airborne phase, add extra height
                extra_height = (state.base_height - 0.83) * 1.5  # Amplify jump effect
                target_foot_z = foot_radius + extra_height
            
            height_adjustment = foot_z - target_foot_z
            new_base_height = 1.5 - height_adjustment
            
            # Set final position
            self.data.qpos[2] = new_base_height
        else:
            # Fallback to simple method
            self.data.qpos[2] = state.base_height
            
        # Final physics update
        mujoco.mj_forward(self.model, self.data)
        
    def run(self):
        """Run visualization"""
        print("\n" + "="*50)
        print("Reference Motion Visualization")
        print("="*50)
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset")
        print("  +/-: Speed up/slow down")
        print("  ESC: Exit")
        print()
        
        # Set initial pose
        mujoco.mj_resetData(self.model, self.data)
        state = self.motion.get_state(0)
        self.set_robot_pose(state)
        
        last_time = time.time()
        
        with viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as v:
            # Set camera
            v.cam.distance = 2.0
            v.cam.elevation = -15
            v.cam.azimuth = 135
            
            while v.is_running():
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Update motion time
                if not self.paused:
                    self.motion_time += dt * self.speed
                    
                # Get current state and set pose
                state = self.motion.get_state(self.motion_time)
                self.set_robot_pose(state)
                
                # Sync display
                v.sync()
                
                # Control frame rate
                time.sleep(0.01)
                
        print("\nVisualization ended")


def main():
    parser = argparse.ArgumentParser(description='Reference motion visualization')
    parser.add_argument('--motion', '-m', type=str, default='jump',
                        help='Motion name (jump/hop)')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='Playback speed (default: 1.0)')
    parser.add_argument('--robot', '-r', type=str, default='PF_TRON1A',
                        help='Robot type (default: PF_TRON1A)')
    
    args = parser.parse_args()
    
    # Build paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    motion_file = os.path.join(script_dir, 'motions', f'{args.motion}.yaml')
    robot_xml = os.path.join(parent_dir, 'pointfoot-mujoco-sim', 'robot-description', 
                             'pointfoot', args.robot, 'xml', 'robot.xml')
    
    # Check files
    if not os.path.exists(motion_file):
        print(f"Error: Motion file not found {motion_file}")
        print("Available motions: jump, hop")
        sys.exit(1)
        
    if not os.path.exists(robot_xml):
        print(f"Error: Robot model not found {robot_xml}")
        sys.exit(1)
        
    # Run visualization
    viz = MotionVisualizer(motion_file, robot_xml, args.speed)
    viz.run()


if __name__ == '__main__':
    main()
