"""
Wrapper script to run octopus.
"""
__author__ = 'ryanquinnnelson'

import sys
import os
import configparser

from octopus.octopus import Octopus

from customized.datasets import DatasetHandler
from customized.models import ModelHandler
from customized.phases import PhaseHandler
from customized.optimizers import OptimizerHandler
from customized.schedulers import SchedulerHandler

# execute before loading torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # better error tracking from gpu


def main():

    # get filename from arguments
    config_file = None
    for arg in sys.argv:
        if '--filename=' in arg:
            config_file = arg.strip().split('=')[1].strip()

    # parse config file
    config_contents = configparser.ConfigParser()
    config_contents.read(config_file)

    # initialize handlers for customized components
    idh = DatasetHandler(config_contents['data']['data_dir'])
    ph = PhaseHandler()
    mh = ModelHandler()
    oh = OptimizerHandler()
    sh = SchedulerHandler()

    # run octopus
    octopus = Octopus(config_file, config_contents, idh, ph, mh, oh, sh)
    octopus.run()


if __name__ == "__main__":
    # execute only if run as a script
    main()
