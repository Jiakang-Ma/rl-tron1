import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg as BaseCommandsCfg

from bipedal_locomotion.tasks.locomotion import mdp

##################
# Scene Definition
##################


@configclass
class PFSceneCfg(InteractiveSceneCfg):
    """Configuration for the test scene"""

    # Terrain configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",      # Terrain path in scene
        terrain_type="plane",           # Terrain type: plane
        terrain_generator=None,         # No terrain generator used
        max_init_terrain_level=0,       # Maximum initial terrain difficulty level
        collision_group=-1,             # Collision group ID

        # Physics material properties
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",    # Friction combine mode: multiply
            restitution_combine_mode="multiply", # Restitution combine mode: multiply
            static_friction=1.0,                # Static friction coefficient
            dynamic_friction=1.0,               # Dynamic friction coefficient
            restitution=1.0,                    # Restitution coefficient
        ),

        # Visual material configuration
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",  # Marble texture material path
            project_uvw=True,              # Enable UV projection
            texture_scale=(0.25, 0.25),    # Texture scaling factor
        ),
        debug_vis=False,   # Don't show debug visualization
    )

    # Sky lighting configuration
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # pointfoot robot
    robot: ArticulationCfg = MISSING

    # Height scanner sensor (to be defined in subclasses)
    height_scanner: RayCasterCfg = MISSING

    # Contact force sensor configuration
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # Sensor attachment path
        history_length=4,                     # History data length
        track_air_time=True,                  # Track air time
        update_period=0.0,                    # Update period (0 means every frame)
    )


##############
# MDP Settings
##############


@configclass
class CommandCfg(BaseCommandsCfg):
    # Gait command configuration
    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Command resampling time range (fixed 5s)
        debug_vis=False,                    # No debug visualization
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.5, 2.5),     # Gait frequency range [Hz]
            offsets=(0.5, 0.5),         # Phase offset range [0-1]
            durations=(0.5, 0.5),       # Contact duration range [0-1]
            swing_height=(0.1, 0.2)     # Swing height range [m]
        ),
    )

    """Post-initialization configuration"""
    def __post_init__(self):
        self.base_velocity.asset_name = "robot"          # Associated robot asset name
        self.base_velocity.heading_command = True        # Enable heading commands
        self.base_velocity.debug_vis = True              # Enable debug visualization
        self.base_velocity.heading_control_stiffness = 1.0  # Heading control stiffness
        self.base_velocity.resampling_time_range = (0.0, 5.0)  # Velocity command resampling time
        self.base_velocity.rel_standing_envs = 0.2       # Standing environments ratio
        self.base_velocity.rel_heading_envs = 0.0        # Heading environments ratio
        # Velocity command ranges
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),      # Forward velocity range [m/s]
            lin_vel_y=(-1.0, 1.0),      # Lateral velocity range [m/s]
            ang_vel_z=(-0.5, 0.5),      # Turning angular velocity range [rad/s]
            heading=(-math.pi, math.pi)  # Heading angle range [rad]
        )


@configclass
class ActionsCfg:
    """Action specifications configuration class"""

    # Joint position action configuration
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",                  # Target asset name
        # List of controlled joint names
        joint_names=["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", 
                    "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"],
        scale=0.25,              # Action scaling factor
        use_default_offset=True, # Use default offset
    )


@configclass
class ObservarionsCfg:
    """Observation specifications configuration class"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy network observation group configuration"""

        # Robot base measurements
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,              # Base angular velocity function
            noise=GaussianNoise(mean=0.0, std=0.05),  # Gaussian noise
            clip=(-100.0, 100.0),               # Value clipping range
            scale=0.25,                         # Scaling factor
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,         # Projected gravity function
            noise=GaussianNoise(mean=0.0, std=0.025),  # Noise configuration
            clip=(-100.0, 100.0),               # Clipping range
            scale=1.0,                          # Scaling factor
        )

        # Robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,            # Joint position function
            noise=GaussianNoise(mean=0.0, std=0.01),  # Noise configuration
            clip=(-100.0, 100.0),               # Clipping range
            scale=1.0,                          # Scaling factor
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,                 # Joint velocity function
            noise=GaussianNoise(mean=0.0, std=0.01),  # Noise configuration
            clip=(-100.0, 100.0),               # Clipping range
            scale=0.05,                         # Scaling factor
        )

        # Last action
        last_action = ObsTerm(func=mdp.last_action)

        # Gait-related observations
        gait_phase = ObsTerm(func=mdp.get_gait_phase)  # Gait phase
        gait_command = ObsTerm(
            func=mdp.get_gait_command, 
            params={"command_name": "gait_command"}  # Gait command
        )
        
        def __post_init__(self):
            self.enable_corruption = True      # Enable observation corruption
            self.concatenate_terms = True      # Concatenate all observation terms
    
    @configclass
    class HistoryObsCfg(ObsGroup):
        """History observation group - for storing observation history"""

        # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=0.25,)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=0.05,)

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10          # History length of 10 steps
            self.flatten_history_dim = False  # Don't flatten history dimension

    @configclass
    class CriticCfg(ObsGroup):
        """Critic network observation group - includes privileged information"""

        # Policy observations (same as agent)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        last_action = ObsTerm(func=mdp.last_action)

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        heights = ObsTerm(func=mdp.height_scan,params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        
        # Privileged observations (only visible to critic)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)    # Joint torques
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)          # Joint accelerations
        robot_feet_contact_force = ObsTerm(                          # Foot contact forces
            func=mdp.robot_feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
            },
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)                    # Robot mass
        robot_inertia = ObsTerm(func=mdp.robot_inertia)              # Robot inertia
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)  # Joint stiffness
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)      # Joint damping
        robot_pos = ObsTerm(func=mdp.robot_pos)                      # Robot position
        robot_vel = ObsTerm(func=mdp.robot_vel)                      # Robot velocity
        robot_material_propertirs = ObsTerm(func=mdp.robot_material_properties)  # Material properties
        robot_base_pose = ObsTerm(func=mdp.robot_base_pose)          # Base pose

        def __post_init__(self):
            self.enable_corruption = False     # No noise for privileged information
            self.concatenate_terms = True      # Concatenate all terms

    @configclass
    class CommandsObsCfg(ObsGroup):
        """Commands observation configuration"""
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}  # Velocity commands
        )
    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class EventsCfg:
    """Events configuration class - defines randomization events during training"""
    # i.e. domain randomization

    # Startup events
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,     # Randomize rigid body mass function
        mode="startup",                         # Startup mode
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),  # Target: robot base
            "mass_distribution_params": (-1.0, 3.0),  # Mass distribution parameters [kg]
            "operation": "add",                 # Operation type: add
        },
        is_global_time=False,                   # Don't use global time
        min_step_count_between_reset=0,         # Min steps between resets
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,     # Randomize link mass
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),  # All left-right links
            "mass_distribution_params": (0.8, 1.2),  # Mass scaling range
            "operation": "scale",               # Operation type: scale
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,  # Randomize mass and inertia
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),  # Mass inertia distribution
            "operation": "scale",
        },
    )
    
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,     # Randomize physics material
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),   # Static friction range
            "dynamic_friction_range": (0.7, 0.9),  # Dynamic friction range
            "restitution_range": (0.0, 1.0),       # Restitution range
            "num_buckets": 48,                      # Discretization buckets
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,          # Randomize actuator gains
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (32, 48),   # Stiffness distribution
            "damping_distribution_params": (2.0, 3.0),   # Damping distribution
            "operation": "abs",                     # Absolute value operation
            "distribution": "uniform",              # Uniform distribution
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,         # Randomize center of mass
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Center of mass offset range (x, y, z) [m]
            "com_distribution_params": ((-0.075, 0.075), (-0.05, 0.06), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # Reset events
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,          # Uniform reset root state
        mode="reset",                               # Reset mode
        params={
            # Pose range
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # Velocity range
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,             # Reset joints by scale
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),         # Position perturbation range
            "velocity_range": (0.0, 0.0),          # Velocity range (reset to 0)
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # Interval events
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,  # Stochastic external force disturbance
        mode="interval",                            # Interval mode
        interval_range_s=(0.0, 0.0),               # Interval time range
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            # Force range [N]
            "force_range": {
                "x": (-500.0, 500.0), "y": (-500.0, 500.0), "z": (-0.0, 0.0),
            },
            # Torque range [N·m]
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,                   # Occurrence probability
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class RewardsCfg:
    """Reward terms configuration class - defines RL reward functions"""

    # Termination-related rewards
    keep_balance = RewTerm(
        func=mdp.stay_alive,    # Stay alive reward
        weight=1.0              # Reward weight
    )

    # tracking related rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )

    # Regulation-related rewards
    pen_base_height = RewTerm(
        func=mdp.base_com_height,                   # Base height penalty
        params={"target_height": 0.78},            # Target height 78cm
        weight=-20.0,                               # Negative weight indicates penalty
    )
    
    # Joint-related penalties
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00008)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-03)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-5e-04)
    
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,                # Undesired contacts penalty
        weight=-0.5,
        params={
            # Monitor non-foot contacts
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,                      # Contact force threshold
        },
    )

    pen_action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty,           # Action smoothness penalty
        weight=-0.04
    )
    pen_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,               # Flat orientation L2 penalty
        weight=-10.0
    )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,                     # Foot distance penalty
        weight=-100,
        params={
            "min_feet_distance": 0.115,            # Minimum foot distance
            "feet_links_name": ["foot_[RL]_Link"]  # Foot link names
        }
    )
    
    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,                   # Foot regulation penalty
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "base_height_target": 0.65,            # Base target height
            "foot_radius": 0.03                    # Foot radius
        },
    )

    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel,                  # Foot landing velocity penalty
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot_[RL]_Link"]),
            "foot_radius": 0.03,
            "about_landing_threshold": 0.08         # About to land threshold
        },
    )
    
    
    # Gait reward
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,                        # Gait reward function
        weight=1.0,
        params={
            "tracking_contacts_shaped_force": -2.0,    # Contact force tracking shaping
            "tracking_contacts_shaped_vel": -2.0,      # Contact velocity tracking shaping
            "gait_force_sigma": 25.0,                  # Gait force sigma
            "gait_vel_sigma": 0.25,                    # Gait velocity sigma
            "kappa_gait_probs": 0.05,                  # Gait probability parameter
            "command_name": "gait_command",            # Command name
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions configuration class"""

    # Time out termination
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Base contact termination (robot falls down)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
            "threshold": 1.0                        # Contact force threshold
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum learning configuration class"""

    # Terrain difficulty curriculum
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


########################
# Environment Definition
########################


@configclass
class PFEnvCfg(ManagerBasedRLEnvCfg):
    """Test environment configuration class"""

    # Scene settings
    scene: PFSceneCfg = PFSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post-initialization configuration"""
        self.decimation = 4                         # Control frequency downsampling (50Hz -> 12.5Hz)
        self.episode_length_s = 20.0               # Episode length 20 seconds
        self.sim.render_interval = 2 * self.decimation  # Rendering interval
        
        # Simulation settings
        self.sim.dt = 0.005                        # Simulation timestep 5ms
        self.seed = 42                             # Random seed
        
        # Update sensor update periods
        # Sync all sensors based on smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
