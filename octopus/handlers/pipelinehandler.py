"""
All things related to running a deep learning pipeline.
"""
__author__ = 'ryanquinnnelson'

import logging
import time
import torch.optim as optim


# TODO: add check against early stopping criteria
class PipelineHandler:
    """
    Defines an object that manages the phases of the deep learning pipeline.
    """

    def __init__(self, wandbconnector, devicehandler, checkpointhandler, models, optimizers, schedulers, model_names,
                 optimizer_names, scheduler_names, optimizerhandler, schedulerhandler, training, validation, testing, checkpoint_file, load_from_checkpoint,
                 checkpoint_cadence, num_epochs, n_pretraining_epochs, reset_schedulers_after_pretraining,
                 scheduler_plateau_metric=None):
        """
        Initialize a PipelineHandler object.
        Args:
            reset_schedulers_after_pretraining:
            wandbconnector (WandbConnector): manages connection to wandb
            devicehandler (DeviceHandler): manages torch.device
            checkpointhandler (CheckpointHandler): manages checkpoint saving and loading
            models (Collection[torch.nn.Module]): Collection of models to save.
            optimizers (Collection[torch.optim]): Collection of optimizers to save.
            schedulers (Collection[torch.optim]): Collection of schedulers to save.
            model_names (Collection[String]): Collection of name to use for each model when saving. Length must match that of models.
            optimizer_names (Collection[String]):Collection of name to use for each optimizer when saving. Length must match that of optimizers.
            scheduler_names (Collection[String]):Collection of name to use for each scheduler when saving. Length must match that of schedulers.

            optimizerhandler (OptimizerHandler):Python class which implements the following methods:<br>
               - get_optimizers(models, config) -> (Collection[torch.optim], Collection[String])<br>
                 where the tuple represents (optimizers, optimizer_names). The number and order of optimizers must
                 match the number and order of their corresponding models.<br><br>

            schedulerhandler (SchedulerHandler):Python class which implements the following methods:<br>
               - get_schedulers(optimizers, config) -> (Collection[torch.optim as optim], Collection[String])<br>
                 where the tuple represents (schedulers, scheduler_names). The number and order of schedulers must
                 match the number and order of their corresponding optimizers.<br><br>

            training (Training):Python class which implements the following methods:<br>
            - run_epoch(epoch, num_epochs, models, optimizers) -> Dict<br><br>

            validation (Validation):Python class which implements the following methods:<br>
            - run_epoch(epoch, num_epochs, models) -> Dict<br><br>

            testing (Testing): Python class which implements the following methods:<br>
            - run_epoch(epoch, num_epochs, models) -> Dict<br><br>

            checkpoint_file (str): Fully-qualified filename of checkpoint file to be loaded, if any
            load_from_checkpoint (Boolean): True if model environment should be loaded from a previously saved checkpoint
            checkpoint_cadence (int): Number of training epochs to complete before saving another checkpoint.
            num_epochs (int): Number of epochs to train
            n_pretraining_epochs (int): Number of epochs to perform pretraining
            reset_schedulers_after_pretraining (Boolean): True if schedulers should be reset after pretraining
            scheduler_plateau_metric (str): Name of the metric the scheduler checks during step(), if necessary. Default value is None.
        """
        logging.info(f'Initializing phase handler...')

        self.devicehandler = devicehandler
        self.checkpointhandler = checkpointhandler
        self.wandbconnector = wandbconnector

        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.model_names = model_names
        self.optimizer_names = optimizer_names
        self.scheduler_names = scheduler_names
        self.optimizerhandler = optimizerhandler
        self.schedulerhandler = schedulerhandler

        self.training = training
        self.validation = validation
        self.testing = testing

        self.checkpoint_file = checkpoint_file
        self.load_from_checkpoint = load_from_checkpoint
        self.checkpoint_cadence = checkpoint_cadence
        self.stats = {}
        self.first_epoch = 1
        self.num_epochs = num_epochs
        self.scheduler_plateau_metric = scheduler_plateau_metric
        self.n_pretraining_epochs = n_pretraining_epochs
        self.reset_schedulers_after_pretraining = reset_schedulers_after_pretraining

    def _load_checkpoint(self):
        """
        Load model environment from previous checkpoint. Replace stats dictionary with stats dictionary recovered
        from checkpoint and update first epoch to next epoch value recovered from checkpoint.

        Returns: None

        """
        device = self.devicehandler.get_device()
        checkpoint = self.checkpointhandler.load(self.checkpoint_file, device, self.models, self.optimizers,
                                                 self.schedulers,
                                                 self.model_names, self.optimizer_names, self.scheduler_names)

        # restore stats
        self.stats = checkpoint['stats']

        # set which epoch to start from
        self.first_epoch = checkpoint['next_epoch']

    def _append_stats(self, curr_stats_dict):
        """
        For each epoch stored in the stats dictionary, send all metrics for that epoch to wandb.

        Args:
            curr_stats_dict (Dict): current dictionary of stats

        Returns:

        """

        for key in curr_stats_dict.keys():
            curr_val = curr_stats_dict[key]

            if key in self.stats.keys():
                # append value to collection
                self.stats[key].append(curr_val)
            else:
                # create new collection for stats
                self.stats[key] = [curr_val]

    def _update_scheduler(self, scheduler, curr_stats):
        """
        Perform a single scheduler step. Check scheduler_plateau_metric from current stats if necessary.
        Args:
            scheduler (nn.optim): scheduler to step
            curr_stats (Dictionary): dictionary of latest run stats from which the latest scheduler_plateau_metric value should be
            extracted
        Returns: None
        """
        if type(scheduler).__name__ == 'ReduceLROnPlateau':
            metric_val = curr_stats[self.scheduler_plateau_metric]
            scheduler.step(metric_val)
        else:
            scheduler.step()

    def _reset_components(self):
        # use wandb configs so we can sweep hyperparameters
        config = self.wandbconnector.wandb_config

        # optimizers
        self.optimizers, self.optimizer_names = self.optimizerhandler.reset_optimizers(self.models, config)

        # schedulers
        self.schedulers, self.scheduler_names = self.schedulerhandler.reset_schedulers(self.optimizers, config)

    def _report_previous_stats(self):
        """
        For each epoch stored in the stats dictionary, send all metrics for that epoch to wandb.
        Returns: None
        """
        logging.info('Reporting previous stats...')
        n_stats = len(self.stats[list(self.stats.keys())[0]])  # calculate how many epochs of stats were collected
        for i in range(0, n_stats):
            epoch_stats_dict = dict()
            for key in self.stats.keys():
                epoch_stats_dict[key] = self.stats[key][i]
            self.wandbconnector.log_stats(epoch_stats_dict)

    def process_epochs(self):
        """
        Run models phases for all epochs. Load model from checkpoint first if necessary and submit all previous
        stats to wandb. Save checkpoints according to the specified cadence.

        Returns:

        """

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self._load_checkpoint()

            # submit old stats to wandb to align with other runs
            self._report_previous_stats()

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):

            # record start time
            start = time.time()

            # determine if scheduler must be reset
            if self.reset_schedulers_after_pretraining and epoch == self.n_pretraining_epochs + 1:
                self._reset_components()

            # train
            train_stats = self.training.run_epoch(epoch, self.num_epochs, self.models, self.optimizers)

            # validate
            val_stats = self.validation.run_epoch(epoch, self.num_epochs, self.models)

            # testing
            test_stats = self.testing.run_epoch(epoch, self.num_epochs, self.models)

            # record end time
            end = time.time()

            # combine all stats dictionaries to report
            curr_stats = train_stats
            curr_stats.update(val_stats)
            curr_stats.update(test_stats)
            curr_stats['runtime'] = end - start
            curr_stats['epoch'] = epoch

            # report stats
            self.wandbconnector.log_stats(curr_stats)
            logging.info(f'stats:{curr_stats}')

            # append current stats to all stats for checkpointing
            self._append_stats(curr_stats)

            # update scheduler for each model
            for i, model in enumerate(self.models):
                self._update_scheduler(self.schedulers[i], curr_stats)

            # save model checkpoint
            if epoch % self.checkpoint_cadence == 0:
                self.checkpointhandler.save(self.models, self.optimizers, self.schedulers,
                                            self.model_names, self.optimizer_names, self.scheduler_names,
                                            epoch + 1, self.stats)

            # # check if early stopping criteria is met
            # if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
            #     logging.info('Early stopping criteria is met. Stopping the training process...')
            #     break  # stop running epochs
