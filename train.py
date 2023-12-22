from tqdm import tqdm
import numpy as np
import torch

def inner_loop(policy, optimizer, buffer, meta_batch, task_id, hparams):
    observations, adjs, actions, logprobs, v_olds, advantages, rewards, returns, fts = buffer.sample(meta_batch, batch_size=hparams.inner_batch_size)
    vf_loss, pg_loss, ent_loss = [], [], []
    # Adapt the policy on the current task
    for step in tqdm(range(hparams.adaptation_steps), desc=f'Adapting task {task_id}', ascii=True, leave=False):
        for observation, adj, action, logprob, v_old, advantage, return_ in zip(observations, adjs, actions, logprobs, v_olds, advantages, returns):
            # update new task policy using the sampled trajectories
            # compute likelihood ratio
            if hparams.is_graph:
                v_pred, new_logprobs, new_entropies = policy.evaluate_actions(observation, adj, action)
            else:
                v_pred, new_logprobs, new_entropies = policy.evaluate_actions(observation, action)
            ratio = torch.exp(new_logprobs - logprob)
            # compute surrogate loss
            obj = ratio * advantage
            obj_clip = ratio.clamp(1.0 - hparams.clip_eps, 1.0 + hparams.clip_eps) * advantage
            policy_loss = -torch.min(obj, obj_clip).mean()
            # compute value loss
            if hparams.vf_is_clipped:
                v_pred_clipped = v_old + (v_pred - v_old).clamp(-hparams.clip_eps, hparams.clip_eps)
                v_loss = 0.5 * torch.max((v_pred - return_).pow(2), (v_pred_clipped - return_).pow(2)).mean()
            else:
                v_loss = 0.5 * (v_pred - return_).pow(2).mean()
            vf_loss.append(v_loss.item())
            pg_loss.append(policy_loss.item())
            # compute entropy loss
            entropy_loss = new_entropies.mean()
            ent_loss.append(entropy_loss.item())
            # compute total loss
            loss = policy_loss + hparams.vf_coef * v_loss - hparams.ent_coef * entropy_loss
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
    ent_loss = np.mean(ent_loss)
    return vf_loss, pg_loss, ent_loss, fts, policy

def outer_loop(meta_policy, task_policies, outer_optimizer, hparams):
    # Update the meta policy using reptile
    update_number = hparams.graph_number*hparams.num_task_episodes/hparams.inner_batch_size
    outer_optimizer.zero_grad()
    for i in range(hparams.meta_batch_size):
        for core_param, task_param in zip(meta_policy.parameters(), task_policies[i].parameters()):
            if core_param.grad is None:
                core_param.grad = torch.zeros_like(core_param)
            core_param.grad += (core_param - task_param) / hparams.meta_batch_size / hparams.inner_lr / hparams.adaptation_steps / update_number
    outer_optimizer.step()