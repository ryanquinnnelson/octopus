"""
Performs environment setup for deep learning and runs a deep learning pipeline.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import configparser

import octopus.utilities.configutilities as cu
from octopus.handlers.logginghandler import LoggingHandler
from octopus.handlers.packagehandler import PackageHandler
from octopus.connectors.wandbconnector import WandbConnector
from octopus.handlers.checkpointhandler import CheckpointHandler
from octopus.handlers.outputhandler import OutputHandler
from octopus.handlers.devicehandler import DeviceHandler
from octopus.handlers.optimizerhandler import OptimizerHandler
from octopus.handlers.schedulerhandler import SchedulerHandler
from octopus.handlers.dataloaderhandler import DataLoaderHandler
from octopus.handlers.phasehandler import PhaseHandler

import customized.datasets as datasets
import customized.models as models
import customized.phases as phases

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
        self.packagehandler = None
        self.wandbconnector = None
        self.checkpointhandler = None
        self.outputhandler = None
        self.devicehandler = None
        self.optimizerhandler = None
        self.schedulerhandler = None
        self.dataloaderhandler = None
        self.datasethandler = None
        self.phasehandler = None

        # models and components
        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

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
        logging.info(f'octopus is setting up wandb...')
        # package handler
        self.packagehandler = PackageHandler()

        # install wandb if necessary
        wandb_version = self.config['wandb']['version']
        self.packagehandler.install_package(wandb_version)

        # parse configuration
        wandb_dir = self.config['wandb']['wandb_dir']
        entity = self.config['wandb']['entity']
        run_name = self.config['DEFAULT']['run_name']
        project = self.config['wandb']['project']
        notes = self.config['wandb']['notes']
        tags = cu.to_string_list(self.config['wandb']['tags'])
        mode = self.config['wandb']['mode']

        # get all hyperparameters from different parts of config so wandb can track things that we might want to change
        # and so wandb can perform a hyperparameter sweep
        hyper_dict = dict(self.config['hyperparameters'])
        hyper_dict.update(dict(self.config['model']))
        hyper_dict.update(dict(self.config['dataloader']))
        config = hyper_dict
        config = cu.convert_configs_to_correct_type(config)

        # initialize connector
        self.wandbconnector = WandbConnector(wandb_dir, entity, run_name, project, notes, tags, mode, config)

        # setup
        self.wandbconnector.login()
        self.wandbconnector.initialize_wandb()

        logging.info('octopus has finished setting up wandb.')

    def install_packages(self):
        logging.info(f'octopus is installing packages...')

        # determine if configuration file has any packages to install
        if self.config.has_option('pip', 'packages'):
            packages = self.config['pip']['packages']
            packages_list = cu.to_string_list(packages)
            self.packagehandler.install_packages(packages_list)

        logging.info('octopus has finished installing packages.')

    def setup_environment(self):
        logging.info(f'octopus is setting up the environment...')

        # checkpoints
        checkpoint_dir = self.config['checkpoint']['checkpoint_dir']
        delete_existing_checkpoints = self.config['checkpoint'].getboolean('delete_existing_checkpoints')
        run_name = self.config['DEFAULT']['run_name']
        load_from_checkpoint = self.config['checkpoint'].getboolean('load_from_checkpoint')

        self.checkpointhandler = CheckpointHandler(checkpoint_dir, delete_existing_checkpoints, run_name,
                                                   load_from_checkpoint)
        self.checkpointhandler.setup_checkpoint_directory()

        # output
        output_dir = self.config['output']['output_dir']
        self.outputhandler = OutputHandler(run_name, output_dir)
        self.outputhandler.setup_output_directory()

        # device
        self.devicehandler = DeviceHandler()
        self.devicehandler.set_device()

        logging.info('octopus has finished setting up the environment.')

    def initialize_models(self):
        logging.info(f'octopus is generating the models...')

        # use wandb configs so we can sweep hyperparameters
        config = self.wandbconnector.wandb_config
        self.models = models.get_models(config)

        logging.info(f'octopus finished generating the models.')

    # TODO: allow for possibility of different types of optimizers/schedulers for each model
    def initialize_model_components(self):
        logging.info(f'octopus is generating the model components...')

        # use wandb configs so we can sweep hyperparameters
        config = self.wandbconnector.wandb_config

        self.optimizerhandler = OptimizerHandler()
        self.schedulerhandler = SchedulerHandler()

        # create a separate optimizer and scheduler for each model
        for model in self.models:
            # optimizer
            opt = self.optimizerhandler.get_optimizer(model, config)
            self.optimizers.append(opt)

            sched = self.schedulerhandler.get_scheduler(opt, config)
            self.schedulers.append(sched)

        logging.info(f'octopus finished generating the model components.')

    def load_data(self):
        logging.info(f'octopus is loading the data...')

        # datasets
        train_dataset, val_dataset, test_dataset = datasets.get_datasets(self.config)

        # dataloader
        self.dataloaderhandler = DataLoaderHandler()

        batch_size = self.config['dataloader'].getint('batch_size')
        num_workers = self.config['dataloader'].getint('num_workers')
        pin_memory = self.config['dataloader'].getboolean('pin_memory')
        device = self.devicehandler.get_device()

        self.dataloaderhandler.define_dataloader_args(batch_size, num_workers, pin_memory, device)

        # load data
        train_dl, val_dl, test_dl = self.dataloaderhandler.load_data(train_dataset, val_dataset, test_dataset)
        self.train_loader = train_dl
        self.val_loader = val_dl
        self.test_loader = test_dl

        logging.info(f'octopus is finished loading the data.')

    def setup_phasehandler(self):
        logging.info(f'octopus is loading the phase handler...')

        if self.config.has_option('checkpoint', 'checkpoint_file'):
            checkpoint_file = self.config['checkpoint']['checkpoint_file']
        else:
            checkpoint_file = None

        num_epochs = self.config['hyperparameters'].getint('num_epochs')
        load_from_checkpoint = self.config['checkpoint'].getboolean('load_from_checkpoint')
        model_names = self.config['checkpoint']['model_names']
        optimizer_names = self.config['checkpoint']['optimizer_names']
        scheduler_names = self.config['checkpoint']['scheduler_names']

        # use wandb configs so we can sweep hyperparameters
        wandb_config = self.wandbconnector.wandb_config
        training, validation, testing = phases.get_phases(wandb_config, self.devicehandler, self.outputhandler)

        self.phasehandler = PhaseHandler(num_epochs, self.devicehandler, self.checkpointhandler, self.schedulerhandler,
                                         self.wandbconnector, training, validation, testing, model_names,
                                         optimizer_names, scheduler_names, load_from_checkpoint,
                                         checkpoint_file)

        logging.info(f'octopus is finished loading the phase handler...')

    def run_pipeline(self):
        """
        Run training, validation, and test phases of training for all epochs.
        Note 1:
        Reason behind moving model to device first:
        https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
        Returns: None
        """
        logging.info('octopus is running the pipeline...')

        self.phasehandler.process_epochs(self.models, self.optimizers,
                                         self.schedulers,
                                         self.train_loader, self.val_loader,
                                         self.test_loader)

        logging.info('octopus has finished running the pipeline.')

    def cleanup(self):
        """
        Perform any cleanup steps. Stop wandb logging for the current run to enable multiple runs within a single
        execution of run_octopus.py.
        Returns: None
        """
        logging.info(f'octopus is cleaning up...')
        self.wandbconnector.run.finish()  # finish logging for this run
        logging.info('octopus cleanup complete.')
