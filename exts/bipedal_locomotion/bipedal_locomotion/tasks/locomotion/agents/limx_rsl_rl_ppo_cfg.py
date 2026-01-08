from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg

import os
robot_type = os.getenv("ROBOT_TYPE")  # Get robot type from environment variable

# Isaac Lab original RSL-RL configuration
@configclass
class PFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24        # Steps collected per environment per iteration
    max_iterations = 15000        # Maximum training iterations
    save_interval = 500           # Model saving interval
    experiment_name = "pf_flat"   # Experiment name
    empirical_normalization = False  # Don't use empirical normalization

    # Actor-Critic network configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,              # Initial action noise std
        actor_hidden_dims=[512, 256, 128],  # Actor network hidden dimensions
        critic_hidden_dims=[512, 256, 128], # Critic network hidden dimensions
        activation="elu",                # Activation function type
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # Value function loss coefficient
        use_clipped_value_loss=True,  # Use clipped value function loss
        clip_param=0.2,               # PPO clipping parameter
        entropy_coef=0.01,            # Entropy regularization coefficient
        num_learning_epochs=5,        # Learning epochs per iteration
        num_mini_batches=4,           # Number of mini-batches
        learning_rate=1.0e-3,         # Learning rate
        schedule="adaptive",          # Adaptive learning rate schedule
        gamma=0.99,                   # Discount factor
        lam=0.95,                     # GAE lambda parameter
        desired_kl=0.01,              # Target KL divergence
        max_grad_norm=1.0,            # Gradient clipping threshold
    )


# PF_TRON1A flat terrain training configuration - optimized for specific robot model
@configclass
class PF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000         # Shorter training for flat terrain
    save_interval = 200           # More frequent saving
    experiment_name = "pf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # Use MLP version of PPO algorithm with history observation support
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,   # Observation history length
    )

    # Encoder configuration - for processing history observation information
    encoder = EncoderCfg(
        output_detach=True,       # Detach output to prevent gradient flow
        num_output_dim=3,         # Output dimensions
        hidden_dims=[256, 128],   # Encoder hidden dimensions
        activation="elu",         # Activation function
        orthogonal_init=False,    # Don't use orthogonal initialization
    )

#-----------------------------------------------------------------
@configclass
class SF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "sf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "wf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )
