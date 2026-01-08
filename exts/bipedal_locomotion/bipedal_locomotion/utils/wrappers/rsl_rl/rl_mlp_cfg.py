# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoAlgorithmMlpCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm configuration with MLP support - extends base PPO config for multi-layer perceptron support"""

    # runner_type: str = "OnPolicyRunner"

    obs_history_len: int = 1  # Observation history length - controls timesteps in input sequence


@configclass
class EncoderCfg:
    """Encoder configuration class - neural network encoder for processing complex observation inputs"""
    
    output_detach: bool = True                    # Output detach - prevents gradient backpropagation to encoder
    num_input_dim: int = MISSING                  # Input dimensions - must be specified when used
    num_output_dim: int = 3                       # Output dimensions - encoded feature dimensions
    hidden_dims: list[int] = [256, 128]           # Hidden layer dimensions - defines network architecture
    activation: str = "elu"                       # Activation function - ELU activation
    orthogonal_init: bool = False                 # Orthogonal initialization - whether to use orthogonal weight init


import os
import copy
import torch
def export_mlp_as_onnx(mlp, path, name, input_dim):

    """Export MLP model to ONNX format - for deployment to different platforms
    Args:
        mlp: MLP model to export
        path: Export path
        name: Model filename
        input_dim: Input dimensions
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".onnx")
    model = copy.deepcopy(mlp).to("cpu")
    model.eval()

    dummy_input = torch.randn(input_dim)
    input_names = ["mlp_input"]
    output_names = ["mlp_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)

def export_policy_as_jit(actor_critic, path):
    """Export policy to TorchScript JIT format - for C++ deployment
    
    Args:
        actor_critic: Actor-Critic model
        path: Export path
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
