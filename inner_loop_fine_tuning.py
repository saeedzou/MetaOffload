import os
import json
import math
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from copy import deepcopy
from tqdm import tqdm
from env.mec_offloaing_envs.offloading_env import Resources
from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
from models import GraphSeq2Seq, BaselineSeq2Seq
from buffer import SingleRolloutBufferPPO
from train import inner_loop
from utils import log_metrics


parser = argparse.ArgumentParser(description="A simple argparse example.")
parser.add_argument('-c', '--config', type=str, help='Config path', default='configs/my_config_val.json')
cfg = parser.parse_args()

with open(cfg.config) as f:
    args = json.load(f)

class Config:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

c = Config(args)
device = c.device
np.random.seed(c.seed)
torch.manual_seed(c.seed)

log_path = (f"Eval_"
            f"{c.graph_file_paths[0].split('/')[-2]}"
            f"obs_{c.obs_dim}_"
            f"h_{c.encoder_units}_"
            f"nhl_{c.num_layers}_"
            f"mbs_{c.meta_batch_size}_"
            f"as_{c.adaptation_steps}_"
            f"g_{c.is_graph}_"
            f"gt_{c.graph_type}_"
            f"n_{c.num_iterations}_"
            f"ibs_{c.inner_batch_size}_"
            f"mgn_{c.max_grad_norm}_"
            f"vf_{c.vf_coef}_"
            f"vfclip_{c.vf_is_clipped}_"
            f"epis_{c.num_task_episodes}"
            f"att_{c.is_attention}_"
            f"seed_{c.seed}_"
            f"olr_{c.outer_lr}_"
            f"ilr_{c.inner_lr}_"
            f"mec_{c.mec_process_capable}_"
            f"mob_{c.mobile_process_capable}_"
            f"ul_{c.bandwidth_up}_"
            f"dl_{c.bandwidth_down}_")
logger_path = "./logs/"+log_path+'.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logger_path,
                    filemode='w',
                    format='%(message)s',
                    level=logging.DEBUG,
                    force=True)
if c.wandb:
    import wandb
    wandb.login(key=c.wandb_key)
    wandb.init(project=c.wandb_project,
               name=log_path,
               config=c)
else:
    wandb = None

if c.save:
    if not os.path.exists(os.path.join(c.save_path, log_path)):
        os.makedirs(os.path.join(c.save_path, log_path))

resources = Resources(mec_process_capable=c.mec_process_capable*1024*1024,
                      mobile_process_capable=c.mobile_process_capable*1024*1024,
                      bandwidth_up=c.bandwidth_up,
                      bandwidth_dl=c.bandwidth_down)

env = OffloadingEnvironment(resource_cluster=resources,
                            batch_size=c.graph_number,
                            graph_number=c.graph_number,
                            graph_file_paths=c.graph_file_paths,
                            time_major=False,
                            encoding=c.encoding)

print(f'Average greedy latency: {np.mean(env.greedy_solution()[1]):.4f}')
print(f'Average all local latency: {np.mean(env.get_all_locally_execute_time()):.4f}')
print(f'Average all mec latency: {np.mean(env.get_all_mec_execute_time()):.4f}')

if c.is_graph:
    policy = GraphSeq2Seq(input_dim=c.obs_dim,
                          hidden_dim=c.encoder_units,
                          output_dim=c.action_dim,
                          num_layers=c.num_layers,
                          device=device,
                          is_attention=c.is_attention,
                          graph=c.graph_type).to(device)
else:
    policy = BaselineSeq2Seq(input_dim=c.obs_dim,
                             hidden_dim=c.encoder_units,
                             output_dim=c.action_dim,
                             num_layers=c.num_layers,
                             device=device,
                             is_attention=c.is_attention).to(device)

if args["load"]:
    policy.load_state_dict(torch.load(args["load_path"], map_location=device))

buffer = SingleRolloutBufferPPO(buffer_size=c.graph_number*c.num_task_episodes,
                                discount=c.gamma, 
                                gae_lambda=c.tau, 
                                device=device)
optimizer = torch.optim.Adam(policy.parameters(), lr=c.inner_lr)

for iteration in tqdm(range(c.start_iter, c.num_iterations), leave=False, disable=True):
    task_policies = []
    fts_before, fts_after = [], []
    vf_losses, pg_losses = [], []
    all_rewards, all_returns = [], []
    
    ### Sample trajectories ###
    buffer.reset()
    buffer.collect_episodes(env=env,
                            policy=policy,
                            device=device,
                            task_id=0,
                            is_graph=c.is_graph)
    buffer.process_task()
    
    
    vf_loss, pg_loss, fts, policy = \
        inner_loop(policy=policy, 
                    optimizer=optimizer, 
                    buffer=buffer, 
                    hparams=c)
    vf_losses.append(vf_loss)
    pg_losses.append(pg_loss)
    fts_before.append(fts)
    task_policies.append(policy)

    ### Log metrics ###
    avg_vf_losses = np.mean(vf_losses)
    avg_pg_losses = np.mean(pg_losses)
    avg_rewards = np.mean(buffer.rewards.sum(-1))
    avg_returns = buffer.returns[:, 0].mean().item()
    avg_fts_before = np.mean(np.concatenate(fts_before))
    avg_fts_after = np.mean(np.concatenate(fts_before))
    
    log_metrics(logger=logger,
                iteration=iteration,
                vf_losses=avg_vf_losses,
                pg_losses=avg_pg_losses,
                rewards=avg_rewards,
                returns=avg_returns,
                finish_times_old=avg_fts_before,
                finish_times_new=avg_fts_after,
                wandb=wandb)