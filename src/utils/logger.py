import wandb
import os

class Logger:
    """
    Logger class for tracking experiment progress.
    """
    def __init__(self, config, fold=None):
        self.config = config
        self.use_wandb = config['logging']['use_wandb']
        
        if self.use_wandb:
            exp_name = config['experiment_name']
            if fold is not None:
                exp_name = f"{exp_name}_fold{fold}"
                
            wandb.init(
                project=config['project_name'],
                name=exp_name,
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
