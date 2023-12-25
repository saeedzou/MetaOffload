def log_metrics(logger, iteration, vf_losses, pg_losses, ent_losses, rewards, returns, finish_times_old, finish_times_new, wandb=None):
    """
    Logs the metrics to the console and optionally to wandb.

    Args:
        logger (logging.Logger): The logger object.
        iteration (int): The current iteration.
        vf_losses (numpy.float64): The average value function loss.
        pg_losses (numpy.float64): The average policy gradient loss.
        ent_losses (numpy.float64): The average entropy loss.
        rewards (numpy.float64): The average rewards.
        returns (numpy.float64): The average returns.
        finish_times_old (numpy.float64): The average finish times of the policy before adaptation.
        finish_times_new (numpy.float64): The average finish times of the policy after adaptation.
        wandb (wandb): The wandb object.
    """
    if iteration == 0:
        logger.info("iteration, vf_loss, pg_loss, ent_loss, rewards, returns, finish_times_old, finish_times_new")

    logger.info(f"{iteration}, "
                f"{vf_losses}, "
                f"{pg_losses}, "
                f"{ent_losses}, "
                f"{rewards}, "
                f"{returns}, "
                f"{finish_times_old}, "
                f"{finish_times_new}")

    if wandb is not None:
        wandb.log({'vf_loss': vf_losses,
                   'pg_loss': pg_losses,
                   'ent_loss': ent_losses,
                   'rewards': rewards,
                   'returns': returns,
                   'finish_times_old': finish_times_old,
                   'finish_times_new': finish_times_new})