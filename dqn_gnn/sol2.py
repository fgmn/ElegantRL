import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import summary
import os
from datetime import datetime
from RoadNetEnv import RoadNetEnv
# from hands_on_rl_dqn import *
import time
import torch
from elegantrl_d3qn import AgentD3QN
from elegantrl import Config
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.evaluator import Evaluator

# 获取当前时间戳并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建带有时间戳的日志目录
log_dir = f"logs/d2qn/{current_time}"
os.makedirs(log_dir, exist_ok=True)

# 创建一个 summary writer
writer = summary.create_file_writer(log_dir)

########################## D3QN TRAIN ###########################

buffer_size = int(4e5)
gpu_id = 0
minimal_size = 500

# 创建环境
env = RoadNetEnv()

args = Config(AgentD3QN, RoadNetEnv)
args.if_use_per = False
args.buffer_size = buffer_size
args.horizon_len = args.max_step * 2
args.if_discrete = True
args.state_dim = env.state_dim
args.action_dim = env.action_dim
args.env_name = env.env_name

args.init_before_training()
torch.set_grad_enabled(False)

args.print()


agent = AgentD3QN(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)

# 实例化 ReplayBuffer 对象
buffer = ReplayBuffer(
    max_size=buffer_size,
    state_dim=args.state_dim,
    action_dim=1 if args.if_discrete else args.action_dim,
    gpu_id=gpu_id,
    num_seqs=args.num_envs,
    if_use_per=args.if_use_per,
    args=args
)

# warm up for ReplayBuffer
state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
agent.last_state = state.detach()
buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
buffer.update(buffer_items)  

# 检查初始化后的状态
print(f"ReplayBuffer initialized with max_size: {buffer.max_size}")
print(f"ReplayBuffer device: {buffer.device}")
print(f"ReplayBuffer using PER: {buffer.if_use_per}")

'''init evaluator'''
eval_env = RoadNetEnv()
evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=True)

'''train loop'''
cwd = args.cwd
break_step = args.break_step
horizon_len = args.horizon_len
if_off_policy = args.if_off_policy
if_save_buffer = args.if_save_buffer
del args

if_train = True
while if_train:
    buffer_items = agent.explore_env(env, horizon_len)

    exp_r = buffer_items[2].mean().item()
    if if_off_policy:
        buffer.update(buffer_items)
    else:
        buffer[:] = buffer_items

    torch.set_grad_enabled(True)
    logging_tuple = agent.update_net(buffer)
    torch.set_grad_enabled(False)

    evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple)
    if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

env.close() if hasattr(env, 'close') else None
evaluator.save_training_curve_jpg()
agent.save_or_load_agent(cwd, if_save=True)
if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
    buffer.save_or_load_history(cwd, if_save=True)
########################## D3QN TRAIN ###########################
