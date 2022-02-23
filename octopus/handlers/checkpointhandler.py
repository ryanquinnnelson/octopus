"""
All things related to model checkpoints.
"""
__author__ = 'ryanquinnnelson'

import os
import logging

import torch

import octopus.utilities.fileutilities as fu


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

    def setup_checkpoint_directory(self):
        """
        Set up handler. Delete and recreate checkpoint directory if delete_existing_checkpoints=True, otherwise
        create checkpoint directory.
        Returns: None
        """
        logging.info('Preparing checkpoint directory...')
        if self.delete_existing_checkpoints:
            fu.delete_directory(self.checkpoint_dir)

        fu.create_directory(self.checkpoint_dir)

    def save(self, models, model_names, optimizers, optimizer_names, schedulers, scheduler_names, next_epoch, stats):

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
            checkpoint[model_name] = model.state_dict()

            optimizer_name = optimizer_names[i]
            optimizer = optimizers[i]
            checkpoint[optimizer_name] = optimizer.state_dict()

            scheduler_name = scheduler_names[i]
            scheduler = schedulers[i]
            checkpoint[scheduler_name] = scheduler.state_dict()

        torch.save(checkpoint, filename)

    def load(self, filename, device, models, model_names, optimizers, optimizer_names, schedulers, scheduler_names):

        logging.info(f'Loading checkpoint from {filename}...')
        checkpoint = torch.load(filename, map_location=device)

        # reload saved state for each model, optimizer, scheduler combination
        for i, model in enumerate(models):
            model_name = model_names[i]
            model.load_state_dict(checkpoint[model_name])

            optimizer_name = optimizer_names[i]
            optimizer = optimizers[i]
            optimizer.load_state_dict(checkpoint[optimizer_name])

            scheduler_name = scheduler_names[i]
            scheduler = schedulers[i]
            scheduler.load_state_dict(checkpoint[scheduler_name])

        return checkpoint
