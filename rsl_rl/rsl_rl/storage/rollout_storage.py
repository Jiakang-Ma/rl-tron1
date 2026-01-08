# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np

class RolloutStorage:
    """经验回放存储系统 - 管理PPO算法的经验数据 / Experience replay storage system - manages experience data for PPO algorithm"""
    class Transition:
        """转换数据结构 - 存储单步环境交互的所有信息 / Transition data structure - stores all information from single environment interaction"""
        
        def __init__(self):
            """初始化转换数据结构 / Initialize transition data structure"""
            self.observations = None            # 当前观测 / Current observations
            self.next_observations = None       # 下一步观测 / Next observations
            self.critic_obs = None              # 评价器观测 / Critic observations
            self.observation_history = None     # 观测历史 / Observation history
            self.commands = None                # 命令输入 / Command inputs
            self.actions = None                 # 执行的动作 / Executed actions
            self.rewards = None                 # 获得的奖励 / Received rewards
            self.dones = None                   # 结束标志 / Done flags
            self.values = None                  # 状态价值 / State values
            self.actions_log_prob = None        # 动作对数概率 / Action log probabilities
            self.action_mean = None             # 动作均值 / Action means
            self.action_sigma = None            # 动作标准差 / Action standard deviations
            self.hidden_states = None           # 隐藏状态（RNN用）/ Hidden states (for RNN)

        def clear(self):
            """清空转换数据 / Clear transition data"""
            self.__init__()

    def __init__(
        self,
        num_envs,                   # 环境数量 / Number of environments
        num_transitions_per_env,    # 每环境转换数 / Number of transitions per environment
        obs_shape,                  # 观测形状 / Observation shape
        all_obs_shape,              # 所有观测形状 / All observation shapes
        obs_history_shape,          # 观测历史形状 / Observation history shape
        commands_shape,             # 命令形状 / Commands shape
        actions_shape,              # 动作形状 / Actions shape
        device="cpu",               # 计算设备 / Computing device
    ):
        
        """初始化存储系统 / Initialize storage system
        
        Args:
            num_envs: 并行环境数量 / Number of parallel environments
            num_transitions_per_env: 每个环境的转换数 / Transitions per environment
            obs_shape: 基础观测的形状 / Shape of basic observations
            all_obs_shape: 所有观测的形状 / Shape of all observations
            obs_history_shape: 观测历史的形状 / Shape of observation history
            commands_shape: 命令的形状 / Shape of commands
            actions_shape: 动作的形状 / Shape of actions
            device: 存储设备 / Storage device
        """
        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        # 核心存储缓冲区 / Core storage buffers
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        self.next_observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )

        # 评价器观测（可选）/ Critic observations (optional)
        if all_obs_shape[0] is not None:
            self.critic_obs = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *all_obs_shape,
                device=self.device
            )
        else:
            self.critic_obs = None
        self.observation_history = torch.zeros(
            num_transitions_per_env, num_envs, *obs_history_shape, device=self.device
        )
        self.commands = torch.zeros(
            num_transitions_per_env, num_envs, *commands_shape, device=self.device
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()
        
        # PPO特定的存储 / PPO-specific storage
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # RNN相关存储 / RNN-related storage
        self.saved_hidden_states_a = None  # Actor隐藏状态 / Actor hidden states
        self.saved_hidden_states_c = None  # Critic隐藏状态 / Critic hidden states

        self.step = 0

    def add_transitions(self, transition: Transition):
        """添加转换数据到存储中 / Add transition data to storage
        
        Args:
            transition: 转换数据 / Transition data
        """

        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        # 复制转换数据到缓冲区 / Copy transition data to buffers
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        if self.critic_obs is not None:
            self.critic_obs[self.step].copy_(transition.critic_obs)
        self.observation_history[self.step].copy_(transition.observation_history)
        self.commands[self.step].copy_(transition.commands)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """保存隐藏状态（用于RNN）/ Save hidden states (for RNN)
        
        Args:
            hidden_states: 隐藏状态元组 / Hidden states tuple
        """

        if hidden_states is None or hidden_states == (None, None):
            return
        
        # 将GRU隐藏状态转换为元组以匹配LSTM格式 / Convert GRU hidden states to tuple to match LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

       # 如需要则初始化 / Initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]

        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """使用GAE计算回报和优势 / Compute returns and advantages using GAE
        
        Args:
            last_values: 最后状态的价值 / Values of last states
            gamma: 折扣因子 / Discount factor
            lam: GAE lambda参数 / GAE lambda parameter
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()

            # 计算TD误差 / Calculate TD error
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )

            # 计算GAE / Calculate GAE
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # 计算并标准化优势 / Compute and normalize advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        """获取统计信息 / Get statistics
        
        Returns:
            平均轨迹长度和平均奖励 / Average trajectory length and mean reward
        """
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(
        self,
        num_group,
        num_mini_batches,
        num_epochs=8,
    ):
        
        """生成小批次数据用于训练 / Generate mini-batch data for training
        
        Args:
            num_group: 环境组数 / Number of environment groups
            num_mini_batches: 小批次数量 / Number of mini-batches
            num_epochs: 训练轮数 / Number of epochs
            
        Yields:
            小批次训练数据 / Mini-batch training data
        """

        group_batch_size = num_group * self.num_transitions_per_env
        group_mini_batch_size = group_batch_size // num_mini_batches
        group_indices = torch.randperm(
            num_mini_batches * group_mini_batch_size,
            requires_grad=False,
            device=self.device,
        )
        group_group_idx = torch.arange(0, num_group)
        group_observations = self.observations[:, group_group_idx, :].flatten(0, 1)

        group_critic_obs = self.critic_obs[:, group_group_idx, :].flatten(0, 1)
        group_obs_history = self.observation_history[:, group_group_idx, :].flatten(0, 1)

        group_commands = self.commands[:, group_group_idx, :].flatten(0, 1)
        group_actions = self.actions[:, group_group_idx, :].flatten(0, 1)
        group_values = self.values[:, group_group_idx, :].flatten(0, 1)
        group_returns = self.returns[:, group_group_idx, :].flatten(0, 1)

        group_old_actions_log_prob = self.actions_log_prob[:, group_group_idx, :].flatten(0, 1)
        group_advantages = self.advantages[:, group_group_idx, :].flatten(0, 1)
        group_old_mu = self.mu[:, group_group_idx, :].flatten(0, 1)
        group_old_sigma = self.sigma[:, group_group_idx, :].flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                group_start = i * group_mini_batch_size
                group_end = (i + 1) * group_mini_batch_size
                group_batch_idx = group_indices[group_start:group_end]

                group_obs_batch = group_observations[group_batch_idx]
                obs_batch = group_obs_batch
                group_critic_obs_batch = group_critic_obs[group_batch_idx]
                critic_obs_batch = group_critic_obs_batch
                
                group_obs_history_batch = group_obs_history[group_batch_idx]
                obs_history_batch = group_obs_history_batch

                group_commands_batch = group_commands[group_batch_idx]
                group_actions_batch = group_actions[group_batch_idx]
                actions_batch = group_actions_batch

                group_target_values_batch = group_values[group_batch_idx]
                target_values_batch = group_target_values_batch

                group_returns_batch = group_returns[group_batch_idx]
                returns_batch = group_returns_batch

                group_old_actions_log_prob_batch = group_old_actions_log_prob[group_batch_idx]
                old_actions_log_prob_batch = group_old_actions_log_prob_batch

                group_advantages_batch = group_advantages[group_batch_idx]
                advantages_batch = group_advantages_batch

                group_old_mu_batch = group_old_mu[group_batch_idx]
                old_mu_batch = group_old_mu_batch

                group_old_sigma_batch = group_old_sigma[group_batch_idx]
                old_sigma_batch = group_old_sigma_batch

                yield obs_batch, critic_obs_batch, obs_history_batch, group_obs_history_batch, group_commands_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,

    def encoder_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """生成编码器训练的小批次数据 / Generate mini-batch data for encoder training
        
        Args:
            num_mini_batches: 小批次数量 / Number of mini-batches
            num_epochs: 训练轮数 / Number of epochs
            
        Yields:
            编码器训练的小批次数据 / Mini-batch data for encoder training
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        if self.critic_obs is not None:
            critic_obs = self.critic_obs.flatten(0, 1)
        else:
            critic_obs = observations
        obs_history = self.observation_history.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                next_obs_batch = next_observations[batch_idx]
                critic_obs_batch = critic_obs[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                yield next_obs_batch, critic_obs_batch, obs_history_batch