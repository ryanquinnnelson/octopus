"""
All things related to data reading and writing.
"""
__author__ = 'ryanquinnnelson'

import logging

from torch.utils.data import DataLoader


class DataLoaderHandler:
    """
    Defines an object to handle DataLoader objects.
    """

    def __init__(self):
        """
        Initialize DataLoaderHandler.
        Args:
            batch_size (int): batch size regardless of a GPU or CPU
            num_workers (int): number of workers for use in DataLoader when a GPU is available
            pin_memory (Boolean): True if DataLoader should use pin memory when a GPU is available
        """
        logging.info('Initializing dataloader handler...')

        self.train_args = None
        self.val_args = None
        self.test_args = None

    def define_dataloader_args(self, batch_size, num_workers, pin_memory, device):
        """
        Set DataLoader hyperparameters based on device. Use the same values for validation and test DataLoader
        hyperparameters.
        Args:
            device (torch.device): device on which model will run
        Returns: None
        """
        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            self.train_args = dict(shuffle=True,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   drop_last=True)
        else:
            self.train_args = dict(shuffle=True,
                                   batch_size=batch_size,
                                   drop_last=True)

        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            self.val_args = dict(shuffle=False,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=True)
        else:
            self.val_args = dict(shuffle=False,
                                 batch_size=batch_size,
                                 drop_last=True)

        self.test_args = self.val_args  # test and validation have matching arguments

        logging.info(f'DataLoader settings for training dataset:{self.train_args}')
        logging.info(f'DataLoader settings for validation dataset:{self.val_args}')
        logging.info(f'DataLoader settings for test dataset:{self.test_args}')

    def train_dataloader(self, dataset):
        """
        Obtain DataLoader for training dataset.
        Args:
            dataset (Dataset): defines training data
        Returns: DataLoader
        """
        dl = DataLoader(dataset, **self.train_args)
        return dl

    def val_dataloader(self, dataset):
        """
        Obtain DataLoader for validation dataset.
        Args:
            dataset (Dataset): defines validation data
        Returns: DataLoader
        """
        dl = DataLoader(dataset, **self.val_args)
        return dl

    def test_dataloader(self, dataset):
        """
        Obtain DataLoader for test dataset.
        Args:
            dataset (Dataset): defines validation data
        Returns: DataLoader
        """
        dl = DataLoader(dataset, **self.test_args)
        return dl

    def load_data(self, train_dataset, val_dataset, test_dataset):

        logging.info('Loading data...')

        # DataLoaders
        train_dl = self.train_dataloader(train_dataset)
        val_dl = self.val_dataloader(val_dataset)
        test_dl = self.test_dataloader(test_dataset)

        return train_dl, val_dl, test_dl
