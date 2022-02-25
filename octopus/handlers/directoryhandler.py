"""
All things related to creating and deleting files and directories.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import shutil


class DirectoryHandler:
    """
    Defines an object that manages the creation and deletion of files and directories.
    """

    def __init__(self):
        logging.info(f'Initializing directory handler....')

    def create_directory(self, path):
        """
        Create directory if it does not exist.
        Args:
            path (str): fully qualified path to the directory
        Returns: None
        """

        if os.path.isdir(path):
            logging.info(f'Directory already exists:{path}.')
        else:
            os.mkdir(path)
            logging.info(f'Created directory:{path}.')

    def delete_directory(self, path):
        """
        Delete directory if it exists.
        Args:
            path (str): fully qualified path to the directory
        Returns:None
        """

        if os.path.isdir(path):
            shutil.rmtree(path)
            logging.info(f'Deleted directory:{path}.')
        else:
            logging.info(f'Directory does not exist:{path}.')

    def delete_file(self, path):
        """
        Delete file if it exists.
        Args:
            path (str): fully qualified path to the file
        Returns:None
        """

        if os.path.isfile(path):
            os.remove(path)
            logging.info(f'Deleted file:{path}')
        else:
            logging.info(f'File does not exist:{path}')
