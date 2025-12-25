import wandb
import os

class Logger:
    """
    Logger class for tracking experiment progress.
    """
    def __init__(self, config):
        self.config = config
        self.use_wandb = config['logging']['use_wandb']
        
        if self.use_wandb:
            wandb.init(
                project=config['project_name'],
                name=config['experiment_name'],
                config=config
            )

    def log_metrics(self, metrics, step):
        if self.use_wandb:
            wandb.log(metrics, step=step)
        else:
            print(f"Step {step}: {metrics}")

    def finish(self):
        if self.use_wandb:
            wandb.finish()
