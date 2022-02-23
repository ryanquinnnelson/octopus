"""
Performs environment setup for deep learning and runs a deep learning pipeline.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import sys
import configparser

from octopus.handlers.logginghandler import LoggingHandler

# execute before loading torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # better error tracking from gpu


class Octopus:
    """
    Class that manages the building and training of deep learning models.
    Parses configuration and passes values into other classes.
    """

    def __init__(self, config_file):
        self.config_file = config_file

        # placeholders
        self.config = None

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
        pass

    def install_packages(self):
        pass

    def setup_environment(self):
        pass


