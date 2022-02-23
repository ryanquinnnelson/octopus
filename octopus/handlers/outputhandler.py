"""
All things related to the location where data will be output.
"""
__author__ = 'ryanquinnnelson'

import logging

import octopus.utilities.fileutilities as fu


class OutputHandler:
    """
    Defines object to handle saving model output.
    """

    def __init__(self, run_name, output_dir):
        """
        Initialize OutputHandler.
        Args:
            run_name (str): Name of the run
            output_dir (str): fully qualified path to the directory where output should be written
        """
        logging.info('Initializing output handler...')
        self.run_name = run_name
        self.output_dir = output_dir

    def setup_output_directory(self):
        """
        Perform all setup for output handler. Create output directory.
        Returns: None
        """
        logging.info('Preparing output directory...')
        fu.create_directory(self.output_dir)
