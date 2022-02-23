"""
All things related to schedulers.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.optim as optim


class SchedulerHandler:
    """
    Defines object to initialize schedulers.
    """

    def __init__(self):

        logging.info('Initializing scheduler handler...')
        self.scheduler_plateau_metric = None
        self.scheduler_type = None

    def get_scheduler(self, optimizer, wandb_config):
        """
        Obtain the scheduler based on parameters.
        Args:
            optimizer (nn.optim): Optimizer associated with the scheduler
        Returns: nn.optim Scheduler
        """
        scheduler = None

        if wandb_config.scheduler_type == 'StepLR':
            self.scheduler_type = wandb_config.scheduler_type
            step_size = wandb_config.scheduler_step_size
            verbose = wandb_config.scheduler_verbose

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, verbose=verbose)

        elif wandb_config.scheduler_type == 'ReduceLROnPlateau':
            self.scheduler_type = wandb_config.scheduler_type
            self.scheduler_plateau_metric = wandb_config.scheduler_plateau_metric
            mode = wandb_config.scheduler_mode
            factor = wandb_config.scheduler_factor
            patience = wandb_config.scheduler_patience
            min_lr = wandb_config.scheduler_min_lr
            verbose = wandb_config.scheduler_verbose

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                                             min_lr=min_lr, verbose=verbose)

        logging.info(f'Scheduler initialized:\n{scheduler}\n{scheduler.state_dict()}')
        return scheduler

    def update_scheduler(self, scheduler, curr_stats):
        """
        Perform a single scheduler step.
        Args:
            scheduler (nn.optim): scheduler to step
            curr_stats (Dictionary): dictionary of latest run stats from which the latest scheduler_plateau_metric value should be
            extracted
        Returns: None
        """
        if self.scheduler_type == 'ReduceLROnPlateau':
            metric_val = curr_stats[self.scheduler_plateau_metric]
            scheduler.step(metric_val)
        else:
            scheduler.step()
