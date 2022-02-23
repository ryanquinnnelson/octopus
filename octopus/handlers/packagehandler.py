"""
All things related to installing modules via pip.
"""
__author__ = 'ryanquinnnelson'

import os
import logging
import subprocess


class PackageHandler:
    """
    Defines class to handle pip installations.
    """

    def __init__(self):
        """
        Initialize PipHandler.
        :param packages_list (List): List of package installation commands
        """
        logging.info('Initializing pip installations handler...')

    def install_packages(self, packages_list):
        """
        Perform pip install for each package in the package_list.
        :return: None
        """
        for m in packages_list:
            self.install_package(m)

    def install_package(self, package):
        """
        Install given package command via pip.
        :param package (str): Command to install a package. Can be the name of a package (pytest) or a longer string with flags and version (--upgrade wandb==0.10.8).
        :return: None
        """
        # split package into individual words if necessary
        words = package.strip().split()
        commands = ['pip', 'install'] + words
        logging.info(''.join([c + ' ' for c in commands]))

        # submit list of commands to subprocess
        process = subprocess.Popen(commands,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        logging.info(stdout.decode("utf-8"))
