import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Get current file directory and construct USD model path
current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/PF_TRON1A/PF_TRON1A.usd")

# Define the articulated robot configuration for the biped
POINTFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Rigid body physics properties configuration
            rigid_body_enabled=True,        # Enable rigid body physics
            disable_gravity=False,          # Don't disable gravity
            linear_damping=0.0,             # Linear damping coefficient
            angular_damping=0.0,            # Angular damping coefficient
            max_linear_velocity=1000.0,     # Maximum linear velocity limit
            max_angular_velocity=1000.0,    # Maximum angular velocity limit
            max_depenetration_velocity=1.0, # Maximum depenetration velocity
        ),
        # Articulation root properties configuration
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,       # Enable self-collision detection
            solver_position_iteration_count=4,  # Position solver iteration count
            solver_velocity_iteration_count=4,  # Velocity solver iteration count
        ),
        activate_contact_sensors=True,   # Activate contact sensors
    ),
    # Robot initial state configuration (in radians)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),   # Initial position (x, y, z)
        joint_pos={
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "foot_R_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # Actuator configuration - defines how to control robot joints
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit=300,      # Maximum output torque (N·m)
            velocity_limit=100.0,  # Maximum angular velocity (rad/s)

            # Joint stiffness parameters - controls position tracking accuracy
            stiffness={
                "abad_L_Joint": 40.0,
                "abad_R_Joint": 40.0,
                "hip_L_Joint": 40.0,
                "hip_R_Joint": 40.0,
                "knee_L_Joint": 40.0,
                "knee_R_Joint": 40.0,
            },

            # Joint damping parameters - controls motion smoothness
            damping={
                "abad_L_Joint": 2.5,
                "abad_R_Joint": 2.5,
                "hip_L_Joint": 2.5,
                "hip_R_Joint": 2.5,
                "knee_L_Joint": 2.5,
                "knee_R_Joint": 2.5,
            },
        ),
    },
)
