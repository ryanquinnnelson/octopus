"""
Common utilities.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import shutil


def create_directory(path):
    """
    Creates directory if it does not exist.
    Args:
        path (str): fully qualified path to the directory
    Returns: None
    """

    if os.path.isdir(path):
        logging.info(f'Directory already exists:{path}.')
    else:
        os.mkdir(path)
        logging.info(f'Created directory:{path}.')


def delete_directory(path):
    """
    Deletes directory if it exists.
    Args:
        path (str): fully qualified path to the directory
    Returns:None
    """

    if os.path.isdir(path):
        shutil.rmtree(path)
        logging.info(f'Deleted directory:{path}.')
    else:
        logging.info(f'Directory does not exist:{path}.')


def delete_file(path):
    """
    Deletes file if it exists.
    Args:
        path (str): fully qualified path to the file
    Returns:None
    """

    if os.path.isfile(path):
        os.remove(path)
        logging.info(f'Deleted file:{path}')
    else:
        logging.info(f'File does not exist:{path}')

