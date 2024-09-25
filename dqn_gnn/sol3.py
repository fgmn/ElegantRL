
import os
from GridNetEnv import GridNetEnv
# from hands_on_rl_dqn import *
import time
import torch
from elegantrl_d3qn import AgentD3QN
from elegantrl import Config
from elegantrl_buffer import ReplayBuffer
from elegantrl_evaluator import Evaluator

########################## D3QN TRAIN ###########################

gpu_id = 0

# 创建环境
env = GridNetEnv()

args = Config(AgentD3QN, GridNetEnv)
args.if_use_per = False
args.buffer_size = int(3e5)
args.max_step = env.max_step
args.horizon_len = args.max_step
args.if_discrete = True
args.state_dim = env.state_dim
args.action_dim = env.action_dim
args.env_name = env.env_name
args.eval_per_step = 2000
args.break_step = int(3e5)
args.net_dims = (64, 32)
args.learning_rate = 5e-4
args.batch_size = 512
args.soft_update_tau = 5e-3
args.init_before_training()
torch.set_grad_enabled(False)

args.print()

agent = AgentD3QN(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
agent.act.explore_rate = 0.3

# 实例化 ReplayBuffer 对象
buffer = ReplayBuffer(
    max_size=args.buffer_size,
    state_dim=args.state_dim,
    action_dim=1 if args.if_discrete else args.action_dim,
    num_edges=env.num_edges,
    num_points=env.num_points,  # 增加图的描述参数
    gpu_id=gpu_id,
    num_seqs=args.num_envs,
    if_use_per=args.if_use_per,
    args=args
)

# warm up for ReplayBuffer
# state, action_mask = env.reset()
ary_obs_edges, ary_obs_nodes, ary_obs_pos, ary_mask = env.reset()
# print("Initial State:", state)
# print("Initial Action Mask:", action_mask)

# state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
obs_edges = torch.tensor(ary_obs_edges, dtype=torch.long, device=agent.device).unsqueeze(0)
obs_nodes = torch.tensor(ary_obs_nodes, dtype=torch.float32, device=agent.device).unsqueeze(0)
obs_pos = torch.tensor(ary_obs_pos, dtype=torch.float32, device=agent.device).unsqueeze(0)
action_mask = torch.tensor(ary_mask, dtype=torch.bool, device=agent.device).unsqueeze(0)

# agent.last_state = state.detach()
agent.last_obs_edges = obs_edges.detach()
agent.last_obs_nodes = obs_nodes.detach()
agent.last_obs_pos = obs_pos.detach()
agent.last_action_mask = action_mask.detach()
buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
buffer.update(buffer_items)

# 检查初始化后的状态
print(f"ReplayBuffer initialized with max_size: {buffer.max_size}")
print(f"ReplayBuffer device: {buffer.device}")
print(f"ReplayBuffer using PER: {buffer.if_use_per}")

'''init evaluator'''
eval_env = GridNetEnv()
evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=True)

'''train loop'''
cwd = args.cwd
break_step = args.break_step
horizon_len = args.horizon_len
if_off_policy = args.if_off_policy
if_save_buffer = args.if_save_buffer
del args

if_train = True
total_steps = 0

while if_train:
    buffer_items = agent.explore_env(env, horizon_len)
    total_steps += horizon_len
    # exp_r = buffer_items[2].mean().item()
    exp_r = buffer_items[4].mean().item()
    if if_off_policy:
        buffer.update(buffer_items)
    else:
        buffer[:] = buffer_items

    torch.set_grad_enabled(True)
    logging_tuple = agent.update_net(buffer, total_steps)
    torch.set_grad_enabled(False)

    evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple)
    if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

# env.visualize_best_actions(agent.act)

env.close() if hasattr(env, 'close') else None
evaluator.save_training_curve_jpg()
agent.save_or_load_agent(cwd, if_save=True)
if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
    buffer.save_or_load_history(cwd, if_save=True)
########################## D3QN TRAIN ###########################
