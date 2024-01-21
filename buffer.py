import numpy as np
import torch


class MetaRolloutBuffer:
    def __init__(self, meta_batch_size, buffer_size, discount=0.99, gae_lambda=0.95, device='cpu', normalize_advantage=True):
        self.meta_batch_size = meta_batch_size
        self.meta_buffer = [SingleRolloutBufferPPO(buffer_size, discount, gae_lambda, device, normalize_advantage) for _ in range(meta_batch_size)]
        self.buffer_size = buffer_size
        self.pos = 0
        self.reset()
    
    def reset(self):
        for m in range(self.meta_batch_size):
            self.meta_buffer[m].reset()
        self.pos = 0


    def collect_episodes(self, env, batch_of_tasks, policy, device, is_graph):
        if isinstance(policy, list):
            for i, task in enumerate(batch_of_tasks):
                self.meta_buffer[i].collect_episodes(env, policy[i], device, task, is_graph)
        else:
            for i, task in enumerate(batch_of_tasks):
                self.meta_buffer[i].collect_episodes(env, policy, device, task, is_graph)
    
    def compute_returns(self):
        for m in range(self.meta_batch_size):
            self.meta_buffer[m].compute_returns_task()
    
    def compute_advantage(self):
        for m in range(self.meta_batch_size):
            self.meta_buffer[m].compute_advantage_task()
    
    def process(self):
        for m in range(self.meta_batch_size):
            self.meta_buffer[m].process_task()


class SingleRolloutBufferPPO:
    def __init__(self, buffer_size, discount=0.99, gae_lambda=0.95, device='cpu', normalize_advantage=True):
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.device = device
        self.normalize_advantage = normalize_advantage
        self.buffer_size = buffer_size
        self.pos = 0
        self.reset()

    def reset(self):
        self.observations = []
        self.adj = []
        self.actions = []
        self.logits = []
        self.Vs = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        self.finish_times = []
        self.pos = 0

    def append(self, observation, adj, action, logits, V, reward, finish_time):
        num_data = observation.shape[0]
        self.observations.append(observation)
        self.adj.append(adj)
        self.actions.append(action.cpu().numpy())
        self.logits.append(logits.detach().cpu().numpy())
        self.Vs.append(V.detach().cpu().numpy())
        self.rewards.append(reward)
        self.pos += num_data
        self.finish_times.append(finish_time)

    def process_task(self):
        self.observations = np.concatenate(self.observations)
        self.adj = np.concatenate(self.adj)
        self.actions = np.concatenate(self.actions)
        self.logits = np.concatenate(self.logits)
        self.Vs = np.concatenate(self.Vs)
        self.rewards = np.concatenate(self.rewards)
        self.finish_times = np.concatenate(self.finish_times)
        self.compute_returns_task()
        self.compute_advantage_task()

    def compute_returns_task(self):
        returns = []
        for t in range(self.buffer_size):
            returns.append(self._discount_cumsum(self.rewards[t], self.discount))
        self.returns = np.stack(returns)

    def compute_advantage_task(self):
        values = np.concatenate([self.Vs, np.zeros((self.buffer_size, 1))], axis=-1)
        deltas = self.rewards + self.discount * values[:, 1:] - values[:, :-1]
        advantages = []
        for t in range(self.buffer_size):
            advantages.append(self._discount_cumsum(deltas[t], self.discount * self.gae_lambda))
        self.advantages = np.stack(advantages)
        if self.normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def sample(self, batch_size=None):
        indices = np.random.permutation(self.observations.shape[0])
        observations = torch.from_numpy(self.observations).to(self.device)[indices]
        adj = torch.from_numpy(self.adj).to(self.device)[indices]
        actions = torch.from_numpy(self.actions).to(self.device)[indices]
        logits = torch.from_numpy(self.logits).to(self.device)[indices]
        Vs = torch.from_numpy(self.Vs).to(self.device)[indices]
        advantages = torch.from_numpy(self.advantages).to(self.device)[indices]
        rewards = torch.from_numpy(self.rewards).to(self.device)[indices]
        returns = torch.from_numpy(self.returns).to(self.device)[indices]
        finish_times = self.finish_times
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

    def collect_episodes(self, env, policy, device, task_id, is_graph):
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
            self.append(obs.cpu().numpy(), adj.cpu().numpy(), actions, logits, Vs, reward, finish_time)
    
    def _discount_cumsum(self, x, discount_factor=0.99):
        T = x.shape[0]
        discounted_return = np.zeros_like(x, dtype=float)

        for t in range(T):
            discount = np.power(discount_factor, np.arange(T - t))
            discounted_return[t] = np.sum(x[t:] * discount)

        return discounted_return