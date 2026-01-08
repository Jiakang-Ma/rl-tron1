#!/usr/bin/env python3
"""
参考动作可视化工具
在 MuJoCo 中播放参考动作，验证动作物理可行性

使用方法:
    /Users/majk/Library/Python/3.9/bin/mjpython motion_data/visualize_motion.py
    /Users/majk/Library/Python/3.9/bin/mjpython motion_data/visualize_motion.py --motion hop
    /Users/majk/Library/Python/3.9/bin/mjpython motion_data/visualize_motion.py --speed 0.5

控制说明:
    - SPACE: 暂停/继续
    - R: 重置到起始
    - +/-: 加速/减速
    - ESC: 退出
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer as viewer

# 添加父目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from motion_data.motion_loader import MotionLoader


class MotionVisualizer:
    """参考动作可视化器"""
    
    def __init__(self, motion_file: str, robot_xml: str, speed: float = 1.0):
        """
        初始化可视化器
        
        Args:
            motion_file: 动作 YAML 文件路径
            robot_xml: 机器人 XML 文件路径
            speed: 播放速度倍率
        """
        # 加载动作
        self.motion = MotionLoader(motion_file)
        print(f"\n加载动作: {self.motion.name}")
        print(f"时长: {self.motion.duration}s")
        print(f"循环: {self.motion.loop}")
        
        # 验证动作
        if not self.motion.validate_motion():
            print("\n⚠️  警告: 动作存在超出关节限制的帧!")
        
        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(robot_xml)
        self.data = mujoco.MjData(self.model)
        
        # 播放控制
        self.speed = speed
        self.paused = False
        self.motion_time = 0.0
        
        # 获取关节索引
        self._get_joint_indices()
        
    def _get_joint_indices(self):
        """获取关节在 qpos 中的索引"""
        self.joint_indices = []
        joint_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
        ]
        for name in joint_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx >= 0:
                # qpos 索引 (跳过 base 的 7 个自由度)
                qpos_idx = self.model.jnt_qposadr[idx]
                self.joint_indices.append(qpos_idx)
            else:
                print(f"Warning: Joint {name} not found")
                
    def key_callback(self, keycode):
        """键盘回调"""
        if keycode == ord(' '):
            self.paused = not self.paused
            print("暂停" if self.paused else "继续")
        elif keycode == ord('R') or keycode == ord('r'):
            self.motion_time = 0.0
            print("重置")
        elif keycode == ord('=') or keycode == ord('+'):
            self.speed = min(self.speed * 1.5, 5.0)
            print(f"速度: {self.speed:.2f}x")
        elif keycode == ord('-'):
            self.speed = max(self.speed / 1.5, 0.1)
            print(f"速度: {self.speed:.2f}x")
            
    def set_robot_pose(self, state):
        """设置机器人姿态 (自动计算正确的基座高度)"""
        # 先重置
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置关节角度
        for i, qpos_idx in enumerate(self.joint_indices):
            self.data.qpos[qpos_idx] = state.joint_positions[i]
        
        # 临时设置较高的基座位置来计算 FK
        self.data.qpos[0] = 0  # x
        self.data.qpos[1] = 0  # y  
        self.data.qpos[2] = 1.5  # z (临时高位置)
        self.data.qpos[3] = 1  # quat w
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = 0
        
        # 计算正运动学
        mujoco.mj_forward(self.model, self.data)
        
        # 获取左右脚的位置
        foot_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_collision")
        foot_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_R_collision")
        
        if foot_l_id >= 0 and foot_r_id >= 0:
            foot_l_pos = self.data.geom_xpos[foot_l_id]
            foot_r_pos = self.data.geom_xpos[foot_r_id]
            
            # 取两脚的最低点
            foot_z = min(foot_l_pos[2], foot_r_pos[2])
            
            # 脚底球半径
            foot_radius = 0.032
            
            # 计算需要的基座高度调整
            # 当前脚底最低点应该在 foot_radius 高度（刚好接触地面）
            # 需要的基座高度 = 当前基座高度 - (当前脚底z - 目标脚底z)
            target_foot_z = foot_radius  # 目标：脚底刚好接触地面
            
            # 如果是跳跃阶段（base_height > 0.85），让脚离开地面
            if state.base_height > 0.85:
                # 空中阶段，增加额外高度
                extra_height = (state.base_height - 0.83) * 1.5  # 放大跳跃效果
                target_foot_z = foot_radius + extra_height
            
            height_adjustment = foot_z - target_foot_z
            new_base_height = 1.5 - height_adjustment
            
            # 设置最终位置
            self.data.qpos[2] = new_base_height
        else:
            # 回退到简单方法
            self.data.qpos[2] = state.base_height
            
        # 最终更新物理
        mujoco.mj_forward(self.model, self.data)
        
    def run(self):
        """运行可视化"""
        print("\n" + "="*50)
        print("参考动作可视化")
        print("="*50)
        print("控制说明:")
        print("  SPACE: 暂停/继续")
        print("  R: 重置")
        print("  +/-: 加速/减速")
        print("  ESC: 退出")
        print()
        
        # 设置初始姿态
        mujoco.mj_resetData(self.model, self.data)
        state = self.motion.get_state(0)
        self.set_robot_pose(state)
        
        last_time = time.time()
        
        with viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as v:
            # 设置相机
            v.cam.distance = 2.0
            v.cam.elevation = -15
            v.cam.azimuth = 135
            
            while v.is_running():
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # 更新动作时间
                if not self.paused:
                    self.motion_time += dt * self.speed
                    
                # 获取当前状态并设置姿态
                state = self.motion.get_state(self.motion_time)
                self.set_robot_pose(state)
                
                # 同步显示
                v.sync()
                
                # 控制帧率
                time.sleep(0.01)
                
        print("\n可视化结束")


def main():
    parser = argparse.ArgumentParser(description='参考动作可视化')
    parser.add_argument('--motion', '-m', type=str, default='jump',
                        help='动作名称 (jump/hop)')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='播放速度 (默认: 1.0)')
    parser.add_argument('--robot', '-r', type=str, default='PF_TRON1A',
                        help='机器人类型 (默认: PF_TRON1A)')
    
    args = parser.parse_args()
    
    # 构建路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    motion_file = os.path.join(script_dir, 'motions', f'{args.motion}.yaml')
    robot_xml = os.path.join(parent_dir, 'pointfoot-mujoco-sim', 'robot-description', 
                             'pointfoot', args.robot, 'xml', 'robot.xml')
    
    # 检查文件
    if not os.path.exists(motion_file):
        print(f"错误: 找不到动作文件 {motion_file}")
        print("可用动作: jump, hop")
        sys.exit(1)
        
    if not os.path.exists(robot_xml):
        print(f"错误: 找不到机器人模型 {robot_xml}")
        sys.exit(1)
        
    # 运行可视化
    viz = MotionVisualizer(motion_file, robot_xml, args.speed)
    viz.run()


if __name__ == '__main__':
    main()
