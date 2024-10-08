import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data, Batch
import torch.nn.init as init

"""DQN"""


class QNetBase(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg


class QNet(QNetBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value  # Q values for multiple actions

    def get_action(self, state):
        state = self.state_norm(state)
        if self.explore_rate < torch.rand(1):
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetDuel(QNetBase):  # Dueling DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv = build_mlp(dims=[dims[-1], 1])  # advantage value
        self.net_val = build_mlp(dims=[dims[-1], action_dim])  # Q value

        layer_init_with_orthogonal(self.net_adv[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val(s_enc)  # q value
        q_adv = self.net_adv(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_action(self, state):
        state = self.state_norm(state)
        if self.explore_rate < torch.rand(1):
            s_enc = self.net_state(state)  # encoded state
            q_val = self.net_val(s_enc)  # q value
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwin(QNetBase):  # Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        return q_val  # one group of Q values

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val1 = self.value_re_norm(q_val1)
        q_val2 = self.net_val2(s_enc)  # q value 2
        q_val2 = self.value_re_norm(q_val2)
        return q_val1, q_val2  # two groups of Q values

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q_val)
            action = torch.multinomial(a_prob, num_samples=1)
        return action


def merge_graphs(obs_edge_index, obs_x):
    # 打印原始obs_x的维度
    # print("Original obs_x size:", obs_x.size())
    
    # 检查obs_x是否有batch维度
    if obs_x.dim() == 4:
        # 合并前两个维度
        batch_size, num_envs, num_nodes, num_features = obs_x.size()
        obs_x = obs_x.view(batch_size * num_envs, num_nodes, num_features)
        # print("Reshaped obs_x size:", obs_x.size())
        # 同样需要更新obs_edge_index的形状
        _, _, _, num_edges = obs_edge_index.size()
        obs_edge_index = obs_edge_index.view(batch_size * num_envs, 2, num_edges)
    else:
        batch_size, num_nodes, num_features = obs_x.size()

    # 合并所有节点特征
    x = obs_x.view(-1, num_features)  # 重新塑形为 [batch_size * num_envs * num_nodes, num_features]

    # 累计节点数
    cum_num_nodes = 0
    edge_indices = []
    # 创建一个批次索引列表
    batch = torch.arange(obs_x.size(0)).repeat_interleave(num_nodes).to(obs_x.device)

    # 调整边索引以适应合并后的节点索引
    for i in range(obs_x.size(0)):
        edge_index = obs_edge_index[i] + cum_num_nodes
        edge_indices.append(edge_index)
        cum_num_nodes += num_nodes

    # 合并所有边索引
    edge_index = torch.cat(edge_indices, dim=1)  # 合并到形状为 [2, total_edges]

    # 创建PyTorch Geometric的Data对象
    data = Data(x=x, edge_index=edge_index, batch=batch)
    return data

# TODO: 添加GNN层
class QNetTwinDuel(QNetBase):  # D3QN: Dueling Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        # 初始化 GNN 层
        gnn_layer = geom_nn.GCNConv
        dp_rate=0.1
        dp_rate_linear=0.5
        gnn_hidden_dim = 16
        layers = []
        layers += [
                    gnn_layer(in_channels=4, out_channels=gnn_hidden_dim),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(dp_rate)
                ]
        layers += [
                    gnn_layer(in_channels=gnn_hidden_dim, out_channels=gnn_hidden_dim),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(dp_rate)
                ]
        self.layers = nn.ModuleList(layers)
        
        # for layer in self.layers:
        #     init.kaiming_normal_(layer.weight, nonlinearity='relu')
        # AttributeError: 'GCNConv' object has no attribute 'weight'

        self.head = nn.Sequential(
            # nn.Dropout(dp_rate_linear),
            nn.Linear(gnn_hidden_dim + state_dim, state_dim) # global pool之后和obs_pos concat
        )   # 最后输出一个长度为state_dim的向量

        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv1 = build_mlp(dims=[dims[-1], 1])  # advantage value 1
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_adv2 = build_mlp(dims=[dims[-1], 1])  # advantage value 2
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        # layer_init_with_orthogonal(self.head[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def gnn_forward(self, x, edge_index, batch_idx, obs_pos):
        if obs_pos.dim() == 3:
            batch_size, num_envs, pos_dim = obs_pos.size()
            obs_pos = obs_pos.view(batch_size * num_envs, pos_dim)

        # print('gnn')
        # print(x.shape)
        # print('edge_index.shape:', edge_index.shape)
        edge_index = edge_index.to(torch.long)  # 确保索引为 int64 类型
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        # print(x.shape)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        # print(x.shape)
        # print('obs_pos.shape:', obs_pos.shape)
        x = torch.cat([x, obs_pos], dim=1)
        # print(x.shape)
        x = self.head(x)
        # x = self.head(obs_pos)
        # print(x.shape)
        return obs_pos

    def forward(self, obs_edges, obs_nodes, obs_pos):
        data = merge_graphs(obs_edges, obs_nodes)
        state = self.gnn_forward(data.x, data.edge_index, data.batch, obs_pos)

        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        q_adv = self.net_adv1(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_q1_q2(self, obs_edges, obs_nodes, obs_pos):
        data = merge_graphs(obs_edges, obs_nodes)
        state = self.gnn_forward(data.x, data.edge_index, data.batch, obs_pos)

        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state

        q_val1 = self.net_val1(s_enc)  # q value 1
        q_adv1 = self.net_adv1(s_enc)  # advantage value 1
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1
        q_duel1 = self.value_re_norm(q_duel1)

        q_val2 = self.net_val2(s_enc)  # q value 2
        q_adv2 = self.net_adv2(s_enc)  # advantage value 2
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        q_duel2 = self.value_re_norm(q_duel2)
        return q_duel1, q_duel2  # two dueling Q values

    # def get_action(self, state):
    #     state = self.state_norm(state)
    #     s_enc = self.net_state(state)  # encoded state
    #     q_val = self.net_val1(s_enc)  # q value
    #     if self.explore_rate < torch.rand(1):
    #         action = q_val.argmax(dim=1, keepdim=True)
    #     else:
    #         a_prob = self.soft_max(q_val)
    #         action = torch.multinomial(a_prob, num_samples=1)
    #     return action
    
    def get_action(self, obs_edges, obs_nodes, obs_pos, action_mask=None):
        # obs_edges: (batch, num_envs, 2, num_edges)
        # obs_nodes: (batch, num_envs, num_points, 4)
        # obs_pos: (batch, num_envs, state_dim)
        # or
        # obs_edges: (1, 2, num_edges)
        # obs_nodes: (1, num_points, 4)
        # obs_pos: (1, state_dim)

        # print("obs_edges:", obs_edges.shape)

        data = merge_graphs(obs_edges, obs_nodes)
        state = self.gnn_forward(data.x, data.edge_index, data.batch, obs_pos)

        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value

        # 应用动作掩码
        if action_mask is not None:
             q_val[~action_mask] = float('-inf')  # 将不可用的动作 Q 值设置为负无穷

        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_val)
            # action = torch.multinomial(a_prob, num_samples=1)
            # 选择一个随机的有效动作
            valid_actions = torch.where(action_mask)[1]  # 获取所有有效动作的索引
            action_index = torch.randint(len(valid_actions), (1,))  # 从有效动作中随机选择一个索引
            action = valid_actions[action_index].unsqueeze(0)  # 保持张量形式和维度

        return action

"""Actor (policy network)"""


class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std


class Actor(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.explore_noise_std = 0.1  # standard deviation of exploration action noise

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        state = self.state_norm(state)
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

    def get_action_noise(self, state: Tensor, action_std: float) -> Tensor:
        state = self.state_norm(state)
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_s = build_mlp(dims=[state_dim, *dims], if_raw_out=False)  # network of encoded state
        self.net_a = build_mlp(dims=[dims[-1], action_dim * 2])  # the average and log_std of action

        layer_init_with_orthogonal(self.net_a[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_s(state)  # encoded state
        a_avg = self.net_a(s_enc)[:, :self.action_dim]
        return a_avg.tanh()  # action

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = Normal(a_avg, a_std)
        return dist.rsample().tanh()  # action (re-parameterize)

    def get_action_logprob(self, state):
        state = self.state_norm(state)
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = Normal(a_avg, a_std)
        action = dist.rsample()

        action_tanh = action.tanh()
        logprob = dist.log_prob(a_avg)
        logprob -= (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return action_tanh, logprob.sum(1)


class ActorFixSAC(ActorSAC):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(dims=dims, state_dim=state_dim, action_dim=action_dim)
        self.soft_plus = torch.nn.Softplus()

    def get_action_logprob(self, state):
        state = self.state_norm(state)
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = Normal(a_avg, a_std)
        action = dist.rsample()

        logprob = dist.log_prob(a_avg)
        logprob -= 2 * (math.log(2) - action - self.soft_plus(action * -2))  # fix logprob using SoftPlus
        return action.tanh(), logprob.sum(1)


class ActorPPO(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class ActorDiscretePPO(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.ActionDist = torch.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        a_prob = self.net(state)  # action_prob without softmax
        return a_prob.argmax(dim=1)  # get the indices of discrete action

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))  # action.shape == (batch_size, 1), action.dtype = torch.int
        dist = self.ActionDist(a_prob)
        logprob = dist.log_prob(action.squeeze(1))
        entropy = dist.entropy()
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.long()


"""Critic (value network)"""


class CriticBase(nn.Module):  # todo state_norm, value_norm
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm


class Critic(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 1])

        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values.squeeze(dim=1)  # q value


class CriticTwin(CriticBase):  # shared parameter
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 2])

        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values.mean(dim=1)  # mean Q value

    def get_q_min(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return torch.min(values, dim=1)[0]  # min Q value

    def get_q1_q2(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values[:, 0], values[:, 1]  # two Q values


class CriticEnsemble(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, num_ensembles: int = 4):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.encoder_sa = build_mlp(dims=[state_dim + action_dim, dims[0]])  # encoder of state and action
        self.decoder_qs = [build_mlp(dims=[*dims, 1]) for _ in range(num_ensembles)]  # decoder of Q values

        for dec_q in self.decoder_qs:
            layer_init_with_orthogonal(dec_q[-1], std=0.5)

    def forward(self, state, action):
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state, action):
        state = self.state_norm(state)
        sa_tmp = self.encoder_sa(torch.cat((state, action), dim=1))
        values = torch.concat([dec_q(sa_tmp) for dec_q in self.decoder_qs], dim=1)
        values = self.value_re_norm(values)
        return values  # Q values


class CriticPPO(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value.squeeze(1)  # q value


"""utils"""


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        return torch.cat(
            (x2, self.dense2(x2)), dim=1
        )  # x3  # x2.shape==(-1, lay_dim*4)


class ConvNet(nn.Module):  # pixel-level state encoder
    def __init__(self, inp_dim, out_dim, image_size=224):
        super().__init__()
        if image_size == 224:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 224, 224)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=110
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(96, 128, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(128, 192, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(192, out_dim),  # size==(batch_size, out_dim)
            )
        elif image_size == 112:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 112, 112)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(96, 128, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(128, out_dim),  # size==(batch_size, out_dim)
            )
        else:
            assert image_size in {224, 112}

    def forward(self, x):
        # assert x.shape == (batch_size, inp_dim, image_size, image_size)
        x = x.permute(0, 3, 1, 2)
        x = x / 128.0 - 1.0
        return self.net(x)

    @staticmethod
    def check():
        inp_dim = 3
        out_dim = 32
        batch_size = 2
        image_size = [224, 112][1]
        # from elegantrl.net import Conv2dNet
        net = ConvNet(inp_dim, out_dim, image_size)

        image = torch.ones((batch_size, image_size, image_size, inp_dim), dtype=torch.uint8) * 255
        print(image.shape)
        output = net(image)
        print(output.shape)
