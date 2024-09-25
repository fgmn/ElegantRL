import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor

from elegantrl_AgentBase import AgentBase
from elegantrl_net import QNet, QNetDuel
from elegantrl_net import QNetTwin, QNetTwinDuel
from elegantrl_config import Config
from elegantrl_buffer import ReplayBuffer

class AgentDQN(AgentBase):
    """
    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNet)
        self.cri_class = None  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.act.explore_rate = getattr(args, "explore_rate", 0.35)
        self.explore_rate = self.act.explore_rate
        # Using ϵ-greedy to select uniformly random actions for exploration with `explore_rate` probability.

        '''optimizer'''
        self.actor_parameters = list(self.act.parameters())
        self.critic_parameters = list(self.cri.parameters())
        # to be fixed: gnn参数未被优化
        # self.ac_optimizer = torch.optim.Adam(self.critic_parameters + self.actor_parameters, lr=self.learning_rate, eps=1e-5)
        # self.ac_optimizer = torch.optim.AdamW(self.critic_parameters, lr=self.learning_rate, eps=1e-5)
        self.ac_optimizer = torch.optim.Adam(self.critic_parameters, lr=self.learning_rate, eps=1e-5)

        self.max_train_steps = getattr(args, "break_step", 1e5)


    def lr_decay(self, total_steps):  # Trick: learning rate Decay
        lr_now = self.learning_rate * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now
        
        # er_now = self.explore_rate * (1 - total_steps / self.max_train_steps)
        # self.act.explore_rate = er_now

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            num_envs == 1
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        action_masks = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.bool).to(self.device)

        state = self.last_state  # state.shape == (1, state_dim) for a single env.
        action_mask = self.last_action_mask

        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(action_mask.sum().item(), size=(1, 1)) if if_random else \
                get_action(state, action_mask)  # different
            
            states[t] = state
            action_masks[t] = action_mask

            ary_action = action[0, 0].detach().cpu().numpy()
            ary_state, reward, done, _ , ary_mask = env.step(ary_action)  # next_state
            if done: ary_state, ary_mask = env.reset()
            # 训练中未使用 max_step
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_mask = torch.as_tensor(ary_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
            

        self.last_state = state  # state.shape == (1, state_dim) for a single env.
        self.last_action_mask = action_mask

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones, action_masks

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (num_envs, state_dim) for a vectorized env.

        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(self.num_envs, 1)) if if_random \
                else get_action(state).detach()  # different
            states[t] = state

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer, total_steps: int) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones, action_masks = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            # self.optimizer_update(self.cri_optimizer, obj_critic)
            self.optimizer_update(self.ac_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        self.lr_decay(total_steps)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            #TODO
            next_qs = self.cri_target(next_ss).max(dim=1, keepdim=True)[0].squeeze(1)  # next q_values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states).gather(1, actions.long()).squeeze(1)    #todo: add action mask
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, q_values

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)

            next_qs = self.cri_target(next_ss).max(dim=1, keepdim=True)[0].squeeze(1)  # q values in next step
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states).gather(1, actions.long()).squeeze(1)
        td_errors = self.criterion(q_values, q_labels)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, q_values

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_value = self.act_target(last_state).argmax(dim=1).detach()  # actor is Q Network in DQN style
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns


class AgentDoubleDQN(AgentDQN):
    """
    Double Deep Q-Network algorithm. “Deep Reinforcement Learning with Double Q-learning”. H. V. Hasselt et al.. 2015.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, action_masks = buffer.sample(batch_size)
            # 获取目标网络对下一个状态的Q值预测
            next_q1, next_q2 = self.cri_target.get_q1_q2(next_ss)

            # 应用动作掩码，不可用的动作Q值设为负无穷
            inf_mask = torch.full_like(next_q1, float('-inf'))
            next_q1_masked = torch.where(action_masks, next_q1, inf_mask)
            next_q2_masked = torch.where(action_masks, next_q2, inf_mask)

            # 选择双Q学习中更小的Q值，并从中取最大值作为目标Q值
            next_qs = torch.min(next_q1_masked, next_q2_masked).max(dim=1, keepdim=True)[0].squeeze(1)
            q_labels = rewards + undones * self.gamma * next_qs

            # next_qs = torch.min(*self.cri_target.get_q1_q2(next_ss)).max(dim=1, keepdim=True)[0].squeeze(1)
            # q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in self.act.get_q1_q2(states)]
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices, action_masks = buffer.sample_for_per(batch_size)
            # 获取目标网络对下一个状态的Q值预测
            next_q1, next_q2 = self.cri_target.get_q1_q2(next_ss)

            # 应用动作掩码，不可用的动作Q值设为负无穷
            inf_mask = torch.full_like(next_q1, float('-inf'))
            next_q1_masked = torch.where(action_masks, next_q1, inf_mask)
            next_q2_masked = torch.where(action_masks, next_q2, inf_mask)

            # 选择双Q学习中更小的Q值，并从中取最大值作为目标Q值
            next_qs = torch.min(next_q1_masked, next_q2_masked).max(dim=1, keepdim=True)[0].squeeze(1)
            q_labels = rewards + undones * self.gamma * next_qs
            # next_qs = torch.min(*self.cri_target.get_q1_q2(next_ss)).max(dim=1, keepdim=True)[0].squeeze(1)
            # q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in self.act.get_q1_q2(states)]
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, q1


'''add dueling q network'''


class AgentDuelingDQN(AgentDQN):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)


class AgentD3QN(AgentDoubleDQN):  # Dueling Double Deep Q Network. (D3QN)
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
