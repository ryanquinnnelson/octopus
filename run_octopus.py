"""
Wrapper script to run octopus.
"""
__author__ = 'ryanquinnnelson'


import sys
import logging

from octopus.octopus import Octopus

def main():
    # run octopus using config file found in the path
    config_file = sys.argv[1]

    # run octopus
    octopus = Octopus(config_file)
    octopus.parse_configuration()
    octopus.setup_logging()
    octopus.setup_wandb()
    # octopus.install_packages()
    # octopus.setup_environment()
    # octopus.initialize_pipeline_components()
    # octopus.run_pipeline()
    # octopus.cleanup()


if __name__ == "__main__":
    # execute only if run as a script
    main()
