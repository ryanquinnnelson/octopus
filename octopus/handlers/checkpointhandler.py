"""
All things related to model checkpoints.
"""
__author__ = 'ryanquinnnelson'

import os
import logging

import torch


class CheckpointHandler:
    """
    Defines an object to handle checkpoints.
    """

    def __init__(self, checkpoint_dir, delete_existing_checkpoints, run_name, load_from_checkpoint):
        """
        Initialize CheckpointHandler. Override given value of delete_existing_checkpoints if loading from a previous
        checkpoint is not None.
        Args:
            checkpoint_dir (str): fully-qualified path where checkpoints should be written
            delete_existing_checkpoints (Boolean): True if checkpoint directory should be deleted and recreated
            run_name (str): Name of the current run
            load_from_checkpoint (str): fully-qualified filename of checkpoint file to load
        """
        logging.info('Initializing checkpoint handler...')
        self.checkpoint_dir = checkpoint_dir
        self.delete_existing_checkpoints = delete_existing_checkpoints
        self.run_name = run_name
        self.load_from_checkpoint = load_from_checkpoint

        # override
        if self.load_from_checkpoint:
            logging.info('Overriding delete_existing_checkpoints value. ' +
                         'Existing checkpoints will not be deleted because checkpoint is being loaded for this run.')
            self.delete_existing_checkpoints = False

    # TODO: redesign so you can control which models are saved for a given checkpoint
    def save(self, models, optimizers, schedulers, model_names, optimizer_names, scheduler_names, next_epoch,
             stats):
        """
        Save current model environment to a checkpoint.

        Args:
            models (Collection[torch.nn.Module]): Collection of deep learning models to save.
            optimizers (Collection[torch.optim]):
            schedulers (Collection[torch.optim]):
            model_names (Collection[String]): Collection of name to use for each model when saving. Length must match that of models.
            optimizer_names (Collection[String]):Collection of name to use for each optimizer when saving. Length must match that of optimizers.
            scheduler_names (Collection[String]):Collection of name to use for each scheduler when saving. Length must match that of schedulers.
            next_epoch (int):next epoch to execute if this model is restored
            stats (Dict): dictionary of statistics for all epochs collected during model training to this point

        Returns: None

        """

        # build filename
        filename = os.path.join(self.checkpoint_dir, f'{self.run_name}.checkpoint.{next_epoch - 1}.pt')
        logging.info(f'Saving checkpoint to {filename}...')

        # build state dictionary
        checkpoint = {
            'next_epoch': next_epoch,
            'stats': stats
        }

        # save state for each model, optimizer, scheduler combination
        for i, model in enumerate(models):
            model_name = model_names[i]
            # logging.info(f'model_name:{model_name}')
            checkpoint[model_name] = model.state_dict()

            optimizer_name = optimizer_names[i]
            # logging.info(f'optimizer_name:{optimizer_name}')
            optimizer = optimizers[i]
            checkpoint[optimizer_name] = optimizer.state_dict()

            scheduler_name = scheduler_names[i]
            # logging.info(f'scheduler_name:{scheduler_name}')
            scheduler = schedulers[i]
            checkpoint[scheduler_name] = scheduler.state_dict()

        torch.save(checkpoint, filename)

    # TODO: redesign so you can control which models are loaded from a given checkpoint
    def load(self, filename, device, models, optimizers, schedulers, model_names, optimizer_names,
             scheduler_names):
        """
        Load a previously saved model environment from a checkpoint file, mapping the load based on the device.

        Args:
            filename (str): fully-qualified filename of checkpoint file
            device (torch.device): device on which model was previously running
            models (Collection[torch.nn.Module]): Collection of models to save.
            optimizers (Collection[torch.optim]): Collection of optimizers to save.
            schedulers (Collection[torch.optim]): Collection of schedulers to save.
            model_names (Collection[String]): Collection of name to use for each model when saving. Length must match that of models.
            optimizer_names (Collection[String]):Collection of name to use for each optimizer when saving. Length must match that of optimizers.
            scheduler_names (Collection[String]):Collection of name to use for each scheduler when saving. Length must match that of schedulers.

        Returns: None

        """

        logging.info(f'Loading checkpoint from {filename}...')
        checkpoint = torch.load(filename, map_location=device)

        # reload saved state for each model, optimizer, scheduler combination
        for i, model in enumerate(models):
            model_name = model_names[i]
            # logging.info(f'model_name:{model_name}')
            model.load_state_dict(checkpoint[model_name])

            optimizer_name = optimizer_names[i]
            # logging.info(f'optimizer_name:{optimizer_name}')
            optimizer = optimizers[i]
            optimizer.load_state_dict(checkpoint[optimizer_name])

            scheduler_name = scheduler_names[i]
            # logging.info(f'scheduler_name:{scheduler_name}')
            scheduler = schedulers[i]
            scheduler.load_state_dict(checkpoint[scheduler_name])

        return checkpoint
