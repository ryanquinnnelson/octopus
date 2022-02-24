"""
All things related to optimizers.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.optim as optim


# TODO: Add additional hyperparameters to optimizer options
# TODO: Add additional optimizers as options
class OptimizerHandler:
    """
    Defines object to handle initializing optimizers.
    """

    def __init__(self):
        """
        Initialize OptimizerHandler.
        Args:
            optimizer_type (str): represents the optimizer to construct
            optimizer_kwargs (Dict): dictionary of arguments for use in optimizer initialization
        """
        logging.info('Initializing optimizer handler...')

    def get_optimizers(self, models, wandb_config):

        optimizers = []
        optimizer_names = ['sn_optimizer', 'en_optimizer']
        for model in models:
            opt = self.get_optimizer(model, wandb_config)
            optimizers.append(opt)

        return optimizers, optimizer_names

    def get_optimizer(self, model, wandb_config):
        """
        Obtain the optimizer based on parameters.
        Args:
            model (nn.Module): model optimizer will manage
        Returns: nn.optim optimizer
        """
        opt = None
        if wandb_config.optimizer_type == 'Adam':
            lr = wandb_config.lr
            opt = optim.Adam(model.parameters(), lr=lr)

        elif wandb_config.optimizer_type == 'SGD':
            lr = wandb_config.lr
            opt = optim.SGD(model.parameters(), lr=lr)

        logging.info(f'Optimizer initialized:\n{opt}')
        logging.info(f'LR={opt.state_dict()["param_groups"][0]["lr"]}')  # to ensure function works during training
        return opt
