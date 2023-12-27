from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

def inner_loop(policy, optimizer, buffer, meta_batch, task_id, hparams):
    """
    Executes the inner loop of the meta-training process.

    Args:
        policy (Policy): The policy network.
        optimizer (Optimizer): The optimizer used to update the policy parameters.
        buffer (Buffer): The replay buffer containing the sampled trajectories.
        meta_batch (int): The relative task id based on sampled tasks.
        task_id (int): The ID of the current task.
        hparams (dict): Hyperparameters for the inner loop.

    Returns:
        vf_loss (float): The average value function loss.
        pg_loss (float): The average policy gradient loss.
        fts (list): List of finish times of DAGs.
        policy (Policy): The updated policy network.
    """
    observations, adjs, actions, logits, v_olds, advantages, rewards, returns, fts = buffer.sample(meta_batch, batch_size=hparams.inner_batch_size)
    vf_loss, pg_loss = [], []
    # Adapt the policy on the current task
    for step in tqdm(range(hparams.adaptation_steps), desc=f'Adapting task {task_id}', ascii=True, leave=False):
        for observation, adj, action, old_logit, v_old, advantage, return_ in zip(observations, adjs, actions, logits, v_olds, advantages, returns):
            # update new task policy using the sampled trajectories
            # compute likelihood ratio
            if hparams.is_graph:
                v_pred, new_logit = policy.evaluate_actions(observation, adj, action)
            else:
                v_pred, new_logit = policy.evaluate_actions(observation, action)
            old_logit = F.softmax(old_logit, dim=-1)
            new_logit = F.softmax(new_logit, dim=-1)
            action = action.type(torch.int64)
            new_logit_a = new_logit.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            old_logit_a = old_logit.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(torch.log(new_logit_a) - torch.log(old_logit_a))
            # compute surrogate loss
            obj = ratio * advantage
            obj_clip = ratio.clamp(1.0 - hparams.clip_eps, 1.0 + hparams.clip_eps) * advantage
            policy_loss = -torch.min(obj, obj_clip).mean()
            # compute value loss
            if hparams.vf_is_clipped:
                v_pred_clipped = v_pred + (v_pred - v_old).clamp(-hparams.clip_eps, hparams.clip_eps)
                v_loss = 0.5 * torch.max((v_pred - return_).pow(2), (v_pred_clipped - return_).pow(2)).mean()
            else:
                v_loss = 0.5 * (v_pred - return_).pow(2).mean()
            vf_loss.append(v_loss.item())
            pg_loss.append(policy_loss.item())
            # compute total loss
            loss = policy_loss + hparams.vf_coef * v_loss
            # zero gradient
            optimizer.zero_grad()
            # compute gradient
            loss.backward()
            # gradient clipping
            if hparams.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), hparams.max_grad_norm)
            # update policy parameters
            optimizer.step()
    vf_loss = np.mean(vf_loss)
    pg_loss = np.mean(pg_loss)
    return vf_loss, pg_loss, fts, policy

def outer_loop(meta_policy, task_policies, outer_optimizer, hparams):
    """
    Performs the outer loop of the meta-training process.

    Args:
        meta_policy (torch.nn.Module): The meta policy network.
        task_policies (list): List of task-specific policy networks.
        outer_optimizer (torch.optim.Optimizer): The optimizer for updating the meta policy.
        hparams (Namespace): Hyperparameters for the training process.

    Returns:
        None
    """
    # Update the meta policy using reptile
    update_number = hparams.graph_number * hparams.num_task_episodes / hparams.inner_batch_size
    outer_optimizer.zero_grad()
    for i in range(hparams.meta_batch_size):
        for core_param, task_param in zip(meta_policy.parameters(), task_policies[i].parameters()):
            if core_param.grad is None:
                core_param.grad = torch.zeros_like(core_param)
            core_param.grad += (core_param - task_param) / hparams.meta_batch_size / hparams.inner_lr / hparams.adaptation_steps / update_number
    outer_optimizer.step()