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
    # run octopus using config file found in the path
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)

    # parse configurations for dataset handler
    idh = DatasetHandler(config['data']['data_dir'])
    ph = PhaseHandler()
    mh = ModelHandler()
    oh = OptimizerHandler()
    sh = SchedulerHandler()

    # run octopus
    octopus = Octopus(config_file, config, idh, ph, mh, oh, sh)
    octopus.setup_logging()
    octopus.setup_environment()
    octopus.setup_wandb()
    octopus.load_data()
    octopus.initialize_models()
    octopus.initialize_model_components()
    octopus.setup_pipeline()
    octopus.run_pipeline()
    octopus.cleanup()


if __name__ == "__main__":
    # execute only if run as a script
    main()
