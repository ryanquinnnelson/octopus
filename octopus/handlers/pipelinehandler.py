import logging
import time

'''
self.wandbconnector, self.devicehandler, checkpointhandler,
                                               training_phase, val_phase, test_phase, 
                                               checkpoint_file,load_from_checkpoint, num_epochs
'''


# TODO: check against early stopping criteria
class PipelineHandler:
    def __init__(self, wandbconnector, devicehandler, checkpointhandler,
                 models, optimizers, schedulers, model_names, optimizer_names, scheduler_names,
                 training, validation, testing,
                 checkpoint_file, load_from_checkpoint, checkpoint_cadence,
                 num_epochs, scheduler_plateau_metric=None):
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

    def load_checkpoint(self):
        device = self.devicehandler.get_device()
        checkpoint = self.checkpointhandler.load(self.checkpoint_file, device, self.models, self.optimizers,
                                                 self.schedulers,
                                                 self.model_names, self.optimizer_names, self.scheduler_names)

        # restore stats
        self.stats = checkpoint['stats']

        # set which epoch to start from
        self.first_epoch = checkpoint['next_epoch']

    def append_stats(self, curr_stats_dict):

        for key in curr_stats_dict.keys():
            curr_val = curr_stats_dict[key]

            if key in self.stats.keys():
                # append value to collection
                self.stats[key].append(curr_val)
            else:
                # create new collection for stats
                self.stats[key] = [curr_val]

    def update_scheduler(self, scheduler, curr_stats):
        """
        Perform a single scheduler step.
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

    def report_previous_stats(self):
        """
        For each epoch stored in the stats dictionary, send all metrics for that epoch to wandb.
        Args:
            wandbconnector (WandbConnector): connection to wandb
        Returns: None
        """
        logging.info('Reporting previous stats...')
        n_stats = len(self.stats[list(self.stats.keys())[0]])  # calculate how many epochs of stats were collected
        for i in range(0, n_stats):
            epoch_stats_dict = dict()
            for key in self.stats.keys():
                epoch_stats_dict[key] = self.stats[key][i]
            self.wandbconnector.log_stats(epoch_stats_dict)

    # TODO: turn on test phase
    def process_epochs(self):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self.load_checkpoint()

            # submit old stats to wandb to align with other runs
            self.report_previous_stats()

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):

            # record start time
            start = time.time()

            # train
            train_stats = self.training.run_epoch(epoch, self.num_epochs, self.models, self.optimizers)

            # validate
            val_stats = self.validation.run_epoch(epoch, self.num_epochs, self.models)

            # testing
            # test_stats = self.testing.run_epoch(epoch, self.num_epochs, self.models)
            test_stats = {}

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
            self.append_stats(curr_stats)

            # update scheduler for each model
            for i, model in enumerate(self.models):
                self.update_scheduler(self.schedulers[i], curr_stats)

            # save model checkpoint
            if epoch % self.checkpoint_cadence == 0:
                self.checkpointhandler.save(self.models, self.optimizers, self.schedulers,
                                            self.model_names, self.optimizer_names, self.scheduler_names,
                                            epoch + 1, self.stats)

            # # check if early stopping criteria is met
            # if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
            #     logging.info('Early stopping criteria is met. Stopping the training process...')
            #     break  # stop running epochs