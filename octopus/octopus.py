"""
Performs environment setup for deep learning and runs a deep learning pipeline.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import sys
import configparser

import octopus.utilities.configutilities as cu
from octopus.handlers.logginghandler import LoggingHandler
from octopus.handlers.packagehandler import PackageHandler
from octopus.connectors.wandbconnector import WandbConnector

# execute before loading torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # better error tracking from gpu


class Octopus:
    """
    Class that manages the building and training of deep learning models.
    Parses configuration and passes values into other classes.
    """

    def __init__(self, config_file):
        self.config_file = config_file
        self.packagehandler = PackageHandler()

        # placeholders
        self.config = None
        self.wandbconnector = None

    def parse_configuration(self):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        self.config = config

    def setup_logging(self):
        # parse configuration
        debug_path = self.config['debug']['debug_path']
        run_name = self.config['DEFAULT']['run_name']

        # setup logging
        lh = LoggingHandler(debug_path, run_name)
        lh.setup_logging()
        lh.draw_logo()
        logging.info('Initializing octopus...')

        # log configuration file details now that logging is set up
        logging.info(f'Parsed configuration from {self.config_file}.')

    def setup_wandb(self):
        # parse configuration
        wandb_dir = self.config['wandb']['wandb_dir']
        entity = self.config['wandb']['entity']
        run_name = self.config['DEFAULT']['run_name']
        project = self.config['wandb']['project']
        notes = self.config['wandb']['notes']
        tags = cu.to_string_list(self.config['wandb']['tags'])
        mode = self.config['wandb']['mode']

        # get all hyperparameters from different parts of config so wandb can track things that we might want to change
        hyper_dict = dict(self.config['hyperparameters'])
        hyper_dict.update(dict(self.config['model']))
        hyper_dict.update(dict(self.config['dataloader']))
        config = hyper_dict

        # initialize connector
        self.wandbconnector = WandbConnector(wandb_dir, entity, run_name, project, notes, tags, mode, config)

        # install wandb if necessary
        self.packagehandler.install_package('--upgrade wandb==0.10.8')

        # setup
        self.wandbconnector.login()
        self.wandbconnector.initialize_wandb()

    def install_packages(self):
        pass

    def setup_environment(self):
        pass


