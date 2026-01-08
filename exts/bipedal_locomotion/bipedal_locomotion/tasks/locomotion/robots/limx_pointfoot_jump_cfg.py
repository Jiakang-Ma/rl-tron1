"""
Jump Task Robot Environment Configuration

Jump motion training environment configuration based on Pointfoot robot.
"""

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.jump_env_cfg import JumpEnvCfg, JumpEnvCfg_PLAY


######################
# Jump Task Environment
######################


@configclass
class PFJumpEnvCfg(JumpEnvCfg):
    """Point Foot jump task training environment"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set robot configuration
        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }
        
        # Disable height scanning (blind task)
        self.scene.height_scanner = None
        
        # Also disable height scanning references in observations
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        
        # Adjust domain randomization parameters
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.0)  # Reduce mass randomization
        
        # Set base contact termination condition
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        
        # Update viewport camera settings
        self.viewer.origin_type = "env"
        
        # Disable curriculum learning (not needed for jump task)
        self.curriculum.terrain_levels = None


@configclass
class PFJumpEnvCfg_PLAY(PFJumpEnvCfg):
    """Point Foot jump task play environment"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Reduce number of test environments
        self.scene.num_envs = 32
        
        # Disable observation noise
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        
        # Disable random pushes
        self.events.push_robot = None
        self.events.add_base_mass = None
