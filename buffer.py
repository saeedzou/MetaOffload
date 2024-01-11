import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, meta_batch_size, buffer_size, discount=0.99, gae_lambda=0.95, device='cpu', normalize_advantage=True):
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.device = device
        self.normalize_advantage = normalize_advantage
        self.buffer_size = buffer_size
        self.meta_batch_size = meta_batch_size
        self.pos = 0
        self.reset()

    def reset(self):
        self.observations = [[] for _ in range(self.meta_batch_size)]
        self.adj = [[] for _ in range(self.meta_batch_size)]
        self.actions = [[] for _ in range(self.meta_batch_size)]
        self.logits = [[] for _ in range(self.meta_batch_size)]
        self.Vs = [[] for _ in range(self.meta_batch_size)]
        self.rewards = [[] for _ in range(self.meta_batch_size)]
        self.advantages = [[] for _ in range(self.meta_batch_size)]
        self.returns = [[] for _ in range(self.meta_batch_size)]
        self.finish_times = [[] for _ in range(self.meta_batch_size)]
        self.pos = 0
        

    def append(self, meta_batch, observation, adj, action, logits, V, reward, finish_time):
        num_data = observation.shape[0]
        self.observations[meta_batch].append(observation)
        self.adj[meta_batch].append(adj)
        self.actions[meta_batch].append(action.cpu().numpy())
        self.logits[meta_batch].append(logits.detach().cpu().numpy())
        self.Vs[meta_batch].append(V.detach().cpu().numpy())
        self.rewards[meta_batch].append(reward)
        self.pos += num_data
        self.finish_times[meta_batch].append(finish_time)

    def process(self):
        self.observations = [np.concatenate(obs) for obs in self.observations]
        self.adj = [np.concatenate(adj) for adj in self.adj]
        self.actions = [np.concatenate(act) for act in self.actions]
        self.logits = [np.concatenate(logit) for logit in self.logits]
        self.Vs = [np.concatenate(V) for V in self.Vs]
        self.rewards = [np.concatenate(reward) for reward in self.rewards]
        self.finish_times = [np.concatenate(finish_time) for finish_time in self.finish_times]
        self.compute_returns()
        self.compute_advantage()

    def process_task(self, meta_batch):
        self.observations[meta_batch] = np.concatenate(self.observations[meta_batch])
        self.adj[meta_batch] = np.concatenate(self.adj[meta_batch])
        self.actions[meta_batch] = np.concatenate(self.actions[meta_batch])
        self.logits[meta_batch] = np.concatenate(self.logits[meta_batch])
        self.Vs[meta_batch] = np.concatenate(self.Vs[meta_batch])
        self.rewards[meta_batch] = np.concatenate(self.rewards[meta_batch])
        self.finish_times[meta_batch] = np.concatenate(self.finish_times[meta_batch])
        self.compute_returns_task(meta_batch)
        self.compute_advantage_task(meta_batch)
    
    def compute_returns_task(self, meta_batch):
        returns = []
        for t in range(self.buffer_size):
            returns.append(self._discount_cumsum(self.rewards[meta_batch][t], self.discount))
        self.returns[meta_batch] = np.stack(returns)

    def compute_advantage_task(self, meta_batch):
        values = np.concatenate([self.Vs[meta_batch], np.zeros((self.buffer_size, 1))], axis=-1)
        deltas = self.rewards[meta_batch] + self.discount * values[:, 1:] - values[:, :-1]
        advantages = []
        for t in range(self.buffer_size):
            advantages.append(self._discount_cumsum(deltas[t], self.discount * self.gae_lambda))
        self.advantages[meta_batch] = np.stack(advantages)
        if self.normalize_advantage:
            self.advantages[meta_batch] = (self.advantages[meta_batch] - self.advantages[meta_batch].mean()) / (self.advantages[meta_batch].std() + 1e-8)

    def compute_returns(self):
        for m in range(self.meta_batch_size):
            returns = []
            for t in range(self.buffer_size):
                returns.append(self._discount_cumsum(self.rewards[m][t], self.discount))
            self.returns[m] = np.stack(returns)

    def compute_advantage(self):
        for m in range(self.meta_batch_size):
            values = np.concatenate([self.Vs[m], np.zeros((self.buffer_size, 1))], axis=-1)
            deltas = self.rewards[m] + self.discount * values[:, 1:] - values[:, :-1]
            advantages = []
            for t in range(self.buffer_size):
                advantages.append(self._discount_cumsum(deltas[t], self.discount * self.gae_lambda))
            self.advantages[m] = np.stack(advantages)
            if self.normalize_advantage:
                self.advantages[m] = (self.advantages[m] - self.advantages[m].mean()) / (self.advantages[m].std() + 1e-8)

    def sample(self, meta_batch, batch_size=None):
        indices = np.random.permutation(self.observations[meta_batch].shape[0])
        observations = torch.from_numpy(self.observations[meta_batch]).to(self.device)[indices]
        adj = torch.from_numpy(self.adj[meta_batch]).to(self.device)[indices]
        actions = torch.from_numpy(self.actions[meta_batch]).to(self.device)[indices]
        logits = torch.from_numpy(self.logits[meta_batch]).to(self.device)[indices]
        Vs = torch.from_numpy(self.Vs[meta_batch]).to(self.device)[indices]
        advantages = torch.from_numpy(self.advantages[meta_batch]).to(self.device)[indices]
        rewards = torch.from_numpy(self.rewards[meta_batch]).to(self.device)[indices]
        returns = torch.from_numpy(self.returns[meta_batch]).to(self.device)[indices]
        finish_times = self.finish_times[meta_batch]
        if batch_size is None:
            batch_size = len(indices)
        observations = observations.split(batch_size, dim=0)
        adj = adj.split(batch_size, dim=0)
        actions = actions.split(batch_size, dim=0)
        logits = logits.split(batch_size, dim=0)
        Vs = Vs.split(batch_size, dim=0)
        advantages = advantages.split(batch_size, dim=0)
        rewards = rewards.split(batch_size, dim=0)
        returns = returns.split(batch_size, dim=0)
        return observations, adj, actions, logits, Vs, advantages, rewards, returns, finish_times
    
    def collect_episodes(self, env, policy, device, meta_batch, task_id, is_graph):
        env.set_task(task_id)
        self.pos = 0
        while self.pos < self.buffer_size:
            obs, adj = env.reset()
            obs = torch.from_numpy(obs).to(device)
            adj = torch.from_numpy(adj).to(device)
            if is_graph:
                actions, logits, Vs = policy(obs, adj)
            else:
                actions, logits, Vs = policy(obs)
            _, reward, _, finish_time = env.step(actions.cpu().numpy())
            self.append(meta_batch, obs.cpu().numpy(), adj.cpu().numpy(), actions, logits, Vs, reward, finish_time)

    
    def _discount_cumsum(self, x, discount_factor=0.99):
        T = x.shape[0]
        discounted_return = np.zeros_like(x, dtype=float)

        for t in range(T):
            discount = np.power(discount_factor, np.arange(T - t))
            discounted_return[t] = np.sum(x[t:] * discount)

        return discounted_return