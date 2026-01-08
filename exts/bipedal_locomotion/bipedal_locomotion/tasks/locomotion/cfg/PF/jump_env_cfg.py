"""
Jump Task Environment Configuration (Jump + Stable)

Encourages jumping but must land stably without falling.
"""

from dataclasses import MISSING

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as base_mdp

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import (
    PFEnvCfg,
    PFSceneCfg,
    RewardsCfg,
    TerminationsCfg,
    EventsCfg,
    ActionsCfg,
    ObservarionsCfg,
)


@configclass
class JumpRewardsCfg:
    """Jump task reward configuration
    
    Strategy: 
    1. Jump height reward encourages takeoff
    2. Landing stability reward prevents falling
    3. Maintain moderate penalties for basic posture
    """
    
    # ===== Core imitation reward =====
    motion_imitation = RewTerm(
        func=mdp.MotionImitationReward,
        weight=8.0,
        params={
            "motion_file": "motion_data/motions/jump.yaml",
            "joint_pos_scale": 3.0,
            "base_height_scale": 5.0,
        }
    )
    
    # ===== Jump height reward =====
    jump_height = RewTerm(
        func=mdp.jump_height_reward,
        weight=5.0,
        params={
            "target_height": 1.0,
        }
    )
    
    # ===== Air time reward =====
    air_time = RewTerm(
        func=mdp.air_time_reward,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "target_air_time": 0.2,
        }
    )
    
    # ===== Landing stability reward (KEY!) =====
    landing_stability = RewTerm(
        func=mdp.landing_stability_reward,
        weight=3.0,  # Increased weight!
    )
    
    # ===== Basic survival reward =====
    keep_balance = RewTerm(
        func=mdp.stay_alive,
        weight=1.0,  # Normal weight
    )
    
    # ===== Penalty terms =====
    pen_joint_torque = RewTerm(func=base_mdp.joint_torques_l2, weight=-0.00003)
    pen_joint_accel = RewTerm(func=base_mdp.joint_acc_l2, weight=-1e-07)
    pen_action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.015)
    pen_joint_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-1.0)
    
    # Undesired contacts (body touching ground)
    pen_undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-1.0,  # Strong penalty
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,
        },
    )
    
    # Orientation penalty (prevent excessive tilting)
    pen_flat_orientation = RewTerm(
        func=base_mdp.flat_orientation_l2,
        weight=-3.0,  # Moderate penalty
    )


@configclass 
class JumpEventsCfg(EventsCfg):
    """Jump task events configuration"""
    
    def __post_init__(self):
        self.push_robot.params["probability"] = 0.0


@configclass
class JumpTerminationsCfg(TerminationsCfg):
    """Jump task termination configuration - enhanced version
    
    Key: If falling (body contacts ground), terminate episode
    """
    
    # Timeout termination
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    
    # Terminate when base contacts ground (fall detection)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
            "threshold": 1.0,
        }
    )
    
    # Terminate when base height is too low (lying down detection)
    base_height_low = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.3,  # Below 0.3m considered lying down
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )


@configclass
class JumpEnvCfg(PFEnvCfg):
    """Jump task environment configuration"""
    
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    terminations: JumpTerminationsCfg = JumpTerminationsCfg()
    events: JumpEventsCfg = JumpEventsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0


@configclass
class JumpEnvCfg_PLAY(JumpEnvCfg):
    """Jump task play environment"""
    
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.scene.num_envs = 50
