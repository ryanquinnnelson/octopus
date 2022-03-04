"""
Manages the process of generating and training of deep learning models.
"""
__author__ = 'ryanquinnnelson'

import logging
import torch

import octopus.utilities.configutilities as cu
from octopus.connectors.wandbconnector import WandbConnector
from octopus.handlers.logginghandler import LoggingHandler
from octopus.handlers.directoryhandler import DirectoryHandler
from octopus.handlers.devicehandler import DeviceHandler
from octopus.handlers.dataloaderhandler import DataLoaderHandler
from octopus.handlers.checkpointhandler import CheckpointHandler
from octopus.handlers.pipelinehandler import PipelineHandler


class Octopus:
    """
    Defines an object which manages the process of generating and training of deep learning models.
    """

    def __init__(self, config_file, config, datasethandler, phasehandler, modelhandler, optimizerhandler,
                 schedulerhandler):
        """
        Initialize an Octopus object.

        Args:
            config_file (str): Name of the file in which the configuration is contained.<br><br>

            config (ConfigParser): Configuration for this run. Expected to be in ConfigParser format.<br><br>

            datasethandler (DatasetHandler): Python class which implements the following methods:<br>
                - get_train_dataset(config) -> torch.utils.data.Dataset<br>
                - get_val_dataset(config) -> torch.utils.data.Dataset<br>
                - get_test_dataset(config) -> torch.utils.data.Dataset<br><br>

            phasehandler (PhaseHandler):Python class which implements the following methods:<br>
               - get_train_phase(devicehandler, dataloader, config) -> Training<br>
               - get_val_phase(devicehandler, dataloader, config) -> Validation<br>
               - get_test_phase(devicehandler, dataloader, config, output_dir) -> Testing<br><br>

               Training is a Python class which implements the following methods:<br>
                  - run_epoch(epoch, num_epochs, models, optimizers) -> Dict<br><br>

               Validation is a Python class which implements the following methods:<br>
                  - run_epoch(epoch, num_epochs, models) -> Dict<br><br>

               Testing is a Python class which implements the following methods:<br>
                  - run_epoch(epoch, num_epochs, models) -> Dict<br><br>

            modelhandler (ModelHandler):Python class which implements the following methods:<br>
               - get_models(config) -> (Collection[torch.nn.Module], Collection[String])<br>
                 where the tuple represents (models, model_names).<br><br>

            optimizerhandler (OptimizerHandler):Python class which implements the following methods:<br>
               - get_optimizers(models, config) -> (Collection[torch.optim], Collection[String])<br>
                 where the tuple represents (optimizers, optimizer_names). The number and order of optimizers must
                 match the number and order of their corresponding models.<br><br>

            schedulerhandler (SchedulerHandler):Python class which implements the following methods:<br>
               - get_schedulers(optimizers, config) -> (Collection[torch.optim as optim], Collection[String])<br>
                 where the tuple represents (schedulers, scheduler_names). The number and order of schedulers must
                 match the number and order of their corresponding optimizers.<br><br>
        """
        # configuration
        self.config_file = config_file
        self.config = config

        # customized
        self.datasethandler = datasethandler
        self.phasehandler = phasehandler
        self.modelhandler = modelhandler
        self.optimizerhandler = optimizerhandler
        self.schedulerhandler = schedulerhandler
        self.output_dir = self.config['output']['output_dir']

        # fixed
        self.wandbconnector = None
        self.devicehandler = None
        self.pipelinehandler = None

        # data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # models and components
        self.models = []
        self.model_names = []
        self.optimizers = []
        self.optimizer_names = []
        self.schedulers = []
        self.scheduler_names = []

    def _setup_logging(self):
        """
        Set up logging to output logs to a specific file in the debug path as well as to the command line, then
        log the logo for the octopus library as well as the configuration file.

        Returns: None

        """
        # parse configuration
        debug_path = self.config['debug']['debug_path']
        run_name = self.config['DEFAULT']['run_name']

        # setup logging
        logginghandler = LoggingHandler(debug_path, run_name)
        logginghandler.setup_logging()
        logginghandler.draw_logo()
        logging.info('Initializing octopus...')

        # log configuration file details now that logging is set up
        logging.info(f'Parsed configuration from {self.config_file}.')

    def _setup_environment(self):
        """
        Create all necessary directories, deleting the old checkpoint directory if specified, and
        define the cpu/gpu device.

        Returns: None

        """
        logging.info(f'octopus is setting up the environment...')
        logging.info(f'octopus is setting up directories...')
        directoryhandler = DirectoryHandler()

        # wandb directory
        wandb_dir = self.config['wandb']['wandb_dir']
        directoryhandler.create_directory(wandb_dir)

        # checkpoint directory
        checkpoint_dir = self.config['checkpoint']['checkpoint_dir']
        delete_existing_checkpoints = self.config['checkpoint'].getboolean('delete_existing_checkpoints')
        if delete_existing_checkpoints:
            directoryhandler.delete_directory(checkpoint_dir)

        directoryhandler.create_directory(checkpoint_dir)

        # output directory
        directoryhandler.create_directory(self.output_dir)

        # define device
        self.devicehandler = DeviceHandler()

        logging.info('octopus has finished setting up the environment.')

    def _load_data(self):
        """
        Load training, validation, and test datasets, then initialize a Dataloader for each.

        Returns: None

        """
        logging.info(f'octopus is loading the data...')

        # datasets
        train_dataset = self.datasethandler.get_train_dataset(self.config)
        val_dataset = self.datasethandler.get_val_dataset(self.config)
        test_dataset = self.datasethandler.get_test_dataset(self.config)

        # dataloader
        batch_size = self.wandbconnector.wandb_config.batch_size
        num_workers = self.config['dataloader'].getint('num_workers')
        pin_memory = self.config['dataloader'].getboolean('pin_memory')
        device = self.devicehandler.get_device()
        dataloaderhandler = DataLoaderHandler(batch_size, num_workers, pin_memory, device)

        # load data
        train_dl, val_dl, test_dl = dataloaderhandler.load_data(train_dataset, val_dataset, test_dataset)
        self.train_loader = train_dl
        self.val_loader = val_dl
        self.test_loader = test_dl

        logging.info(f'octopus is finished loading the data.')

    def _setup_wandb(self):
        """
        Build a dictionary of all parameters wandb should track, then initialize wandb using specified configurations.
        Initializing wandb sets up a run on the wandb website to track the training for this model.

        Returns: None

        """
        logging.info(f'octopus is setting up wandb...')
        # parse configuration
        wandb_dir = self.config['wandb']['wandb_dir']
        entity = self.config['wandb']['entity']
        run_name = self.config['DEFAULT']['run_name']
        project = self.config['wandb']['project']
        notes = self.config['wandb']['notes']
        tags = cu.to_string_list(self.config['wandb']['tags'])
        mode = self.config['wandb']['mode']

        # get all hyperparameters from different sections of config
        # so wandb can track things that we might want to change
        # and so wandb can perform a hyperparameter sweep
        sections = cu.to_string_list(self.config['wandb']['config_sections_to_track'])
        config = {}
        for s in sections:
            config.update(dict(self.config[s]))
        config = cu.convert_configs_to_correct_type(config)

        # initialize connector
        self.wandbconnector = WandbConnector(wandb_dir, entity, run_name, project, notes, tags, mode, config)

        # setup
        self.wandbconnector.initialize_wandb()

        logging.info('octopus has finished setting up wandb.')

    def _initialize_models(self):
        """
        Generate models using the model handler, move the models to the gpu if specified, and track the models with
        wandb.

        Note 1:
        Reason behind moving model to device first:
        https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least

        Returns: None

        """
        logging.info(f'octopus is generating the models...')

        # use wandb configs so we can sweep hyperparameters
        config = self.wandbconnector.wandb_config
        self.models, self.model_names = self.modelhandler.get_models(config)

        moved_models = []
        for model in self.models:
            # move model if necessary
            m = self.devicehandler.move_model_to_device(model)  # move before optimizer init - Note 1
            moved_models.append(m)

        # replace collection of models with moved version
        self.models = moved_models

        # if self.devicehandler.device.type == 'cuda':
        #     # TODO: parameterize this functionality to make back into generic framework
        #     self.models[1](torch.ones(2, 4, 224, 332).cuda(), 0)  # dummy run for EN to initialize any lazy parameters
        #     logging.info(f'EN after dummy:{self.models[1]}')

        for model in self.models:

            # track model
            self.wandbconnector.watch(model)

        logging.info(f'octopus finished generating the models.')

    def _initialize_model_components(self):
        """
        Generate an optimizer for each model, then generate a scheduler for each optimizer.

        Returns: None

        """
        logging.info(f'octopus is generating the model components...')

        # use wandb configs so we can sweep hyperparameters
        config = self.wandbconnector.wandb_config

        # optimizers
        self.optimizers, self.optimizer_names = self.optimizerhandler.get_optimizers(self.models, config)

        # schedulers
        self.schedulers, self.scheduler_names = self.schedulerhandler.get_schedulers(self.optimizers, config)

        logging.info(f'octopus finished generating the model components.')

    def _setup_pipeline(self):
        """
        Perform all tasks to set up the components of the deep learning pipeline. Initialize the checkpoint handler,
        generate the training, validation, and test phases using the phase handler, and extract all parameters that
        must be supplied to the phase handler.

        Returns:

        """
        logging.info(f'octopus is setting up the pipeline...')

        # use wandb configs so we can sweep hyperparameters
        config = self.wandbconnector.wandb_config

        # checkpointing
        checkpoint_dir = self.config['checkpoint']['checkpoint_dir']
        delete_existing_checkpoints = self.config['checkpoint'].getboolean('delete_existing_checkpoints')
        run_name = self.config['DEFAULT']['run_name']
        load_from_checkpoint = self.config['checkpoint'].getboolean('load_from_checkpoint')

        checkpointhandler = CheckpointHandler(checkpoint_dir, delete_existing_checkpoints, run_name,
                                              load_from_checkpoint)

        # phases
        logging.info(f'Initializing phases...')
        training_phase = self.phasehandler.get_train_phase(self.devicehandler, self.train_loader, config)
        val_phase = self.phasehandler.get_val_phase(self.devicehandler, self.val_loader, config)
        test_phase = self.phasehandler.get_test_phase(self.devicehandler, self.test_loader, config, self.output_dir)

        # pipeline handler
        checkpoint_cadence = self.config['checkpoint'].getint('checkpoint_cadence')
        load_from_checkpoint = self.config['checkpoint'].getboolean('load_from_checkpoint')
        num_epochs = self.config['hyperparameters'].getint('num_epochs')

        if self.config.has_option('checkpoint', 'checkpoint_file'):
            checkpoint_file = self.config['checkpoint']['checkpoint_file']
        else:
            checkpoint_file = None

        if self.config.has_option('hyperparameters', 'scheduler_plateau_metric'):
            scheduler_plateau_metric = self.config['hyperparameters']['scheduler_plateau_metric']
        else:
            scheduler_plateau_metric = None

        self.pipelinehandler = PipelineHandler(self.wandbconnector,
                                               self.devicehandler,
                                               checkpointhandler,
                                               self.models,
                                               self.optimizers,
                                               self.schedulers,
                                               self.model_names,
                                               self.optimizer_names,
                                               self.scheduler_names,
                                               training_phase,
                                               val_phase,
                                               test_phase,
                                               checkpoint_file,
                                               load_from_checkpoint,
                                               checkpoint_cadence,
                                               num_epochs,
                                               scheduler_plateau_metric)

        logging.info(f'octopus is finished setting up the pipeline.')

    def _run_pipeline(self):
        """
        Run training, validation, and test phases of training for all epochs.

        Returns: None
        """
        logging.info('octopus is running the pipeline...')

        self.pipelinehandler.process_epochs()

        logging.info('octopus has finished running the pipeline.')

    def _cleanup(self):
        """
        Perform any cleanup steps. Stop wandb logging for the current run to enable multiple runs within a single
        execution of run_octopus.py.

        Returns: None
        """
        logging.info(f'octopus is cleaning up...')
        self.wandbconnector.run.finish()  # finish logging for this run
        logging.info('octopus cleanup complete.')

    def run(self):
        self._setup_logging()
        self._setup_environment()
        self._setup_wandb()
        self._load_data()
        self._initialize_models()
        self._initialize_model_components()
        self._setup_pipeline()
        self._run_pipeline()
        self._cleanup()
