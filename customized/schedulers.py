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

    def get_schedulers(self, optimizers, wandb_config):

        scheduler_names = ['sn_scheduler', 'en_scheduler']
        schedulers = []

        for optimizer in optimizers:
            s = self.get_scheduler(optimizer, wandb_config)
            schedulers.append(s)

        return schedulers, scheduler_names

    def get_scheduler(self, optimizer, wandb_config):
        """
        Obtain the scheduler based on parameters.
        Args:
            optimizer (nn.optim): Optimizer associated with the scheduler
        Returns: nn.optim Scheduler
        """
        scheduler = None

        if wandb_config.scheduler_type == 'StepLR':
            step_size = wandb_config.scheduler_step_size
            verbose = wandb_config.scheduler_verbose

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, verbose=verbose)

        elif wandb_config.scheduler_type == 'ReduceLROnPlateau':
            mode = wandb_config.scheduler_mode
            factor = wandb_config.scheduler_factor
            patience = wandb_config.scheduler_patience
            min_lr = wandb_config.scheduler_min_lr
            verbose = wandb_config.scheduler_verbose

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                                             min_lr=min_lr, verbose=verbose)

        logging.info(f'Scheduler initialized:\n{scheduler}\n{scheduler.state_dict()}')
        return scheduler
