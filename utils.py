import torch.nn as nn

def log_metrics(logger, iteration, vf_losses, pg_losses, rewards, returns, finish_times_old, finish_times_new, wandb=None):
    """
    Logs the metrics to the console and optionally to wandb.

    Args:
        logger (logging.Logger): The logger object.
        iteration (int): The current iteration.
        vf_losses (numpy.float64): The average value function loss.
        pg_losses (numpy.float64): The average policy gradient loss.
        rewards (numpy.float64): The average rewards.
        returns (numpy.float64): The average returns.
        finish_times_old (numpy.float64): The average finish times of the policy before adaptation.
        finish_times_new (numpy.float64): The average finish times of the policy after adaptation.
        wandb (wandb): The wandb object.
    """
    if iteration == 0:
        logger.info("iteration, vf_loss, pg_loss, rewards, returns, finish_times_old, finish_times_new")

    logger.info(f"{iteration}, "
                f"{vf_losses}, "
                f"{pg_losses}, "
                f"{rewards}, "
                f"{returns}, "
                f"{finish_times_old}, "
                f"{finish_times_new}")

    if wandb is not None:
        wandb.log({'vf_loss': vf_losses,
                   'pg_loss': pg_losses,
                   'rewards': rewards,
                   'returns': returns,
                   'finish_times_old': finish_times_old,
                   'finish_times_new': finish_times_new})

def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module

def recurrent_init(module):
    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    return module

class Config:
    def __init__(self, data, mode='ppo'):
        self.mec_process_capable = data['mec_process_capable']
        self.mobile_process_capable = data['mobile_process_capable']
        self.bandwidth_up = data['bandwidth_up']
        self.bandwidth_down = data['bandwidth_down']
        self.outer_lr = data['outer_lr']
        self.inner_lr = data['inner_lr']
        self.seed = data['seed']
        self.meta_batch_size = data['meta_batch_size']
        self.num_iterations = data['num_iterations']
        self.freeze_after = data['freeze_after']
        self.start_iter = data['start_iter']
        self.inner_batch_size = data['inner_batch_size']
        self.max_grad_norm = data['max_grad_norm']
        self.adaptation_steps = data['adaptation_steps']
        self.vf_is_clipped = data['vf_is_clipped']
        self.vf_coef = data['vf_coef']
        self.ent_coef = data['ent_coef']
        self.is_graph = data['is_graph']
        self.graph_type = data['graph_type']
        self.encoding = data['encoding']
        self.obs_dim = data['obs_dim']
        self.action_dim = data['action_dim']
        self.encoder_units = data['encoder_units']
        self.decoder_units = data['decoder_units']
        self.num_layers = data['num_layers']
        self.is_attention = data['is_attention']
        self.device = data['device']
        self.num_task_episodes = data['num_task_episodes']
        self.clip_eps = data['clip_eps']
        self.gamma = data['gamma']
        self.tau = data['tau']
        self.wandb = data['wandb']
        self.wandb_key = data['wandb_key']
        self.wandb_project = data['wandb_project']
        self.wandb_name = data['wandb_name']
        self.log_path = data['log_path']
        self.save = data['save']
        self.save_path = data['save_path']
        self.save_every = data['save_every']
        self.load = data['load']
        self.load_path = data['load_path']
        self.graph_number = data['graph_number']
        self.graph_file_paths = data['graph_file_paths']
        if mode=='ppg':
            self.E_pi = data['E_pi']
            self.N_pi = data['N_pi']
            self.E_v = data['E_v']
            self.E_aux = data['E_aux']
            self.beta_clone = data['beta_clone']