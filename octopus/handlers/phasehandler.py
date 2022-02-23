import logging
import time


# TODO: check against early stopping criteria
class PhaseHandler:
    def __init__(self, num_epochs, devicehandler, checkpointhandler, schedulerhandler, wandbconnector, training,
                 validation, testing, model_names, optimizer_names, scheduler_names,
                 load_from_checkpoint,
                 checkpoint_file=None):
        logging.info(f'Initializing phase handler...')

        self.devicehandler = devicehandler
        self.checkpointhandler = checkpointhandler
        self.wandbconnector = wandbconnector
        self.schedulerhandler = schedulerhandler
        self.training = training
        self.validation = validation
        self.testing = testing
        self.model_names = model_names
        self.optimizer_names = optimizer_names
        self.scheduler_names = scheduler_names

        self.checkpoint_file = checkpoint_file
        self.load_from_checkpoint = load_from_checkpoint
        self.stats = {}
        self.first_epoch = 1
        self.num_epochs = num_epochs

    def load_checkpoint(self, models_list, optimizers, schedulers):
        device = self.devicehandler.get_device()
        checkpoint = self.checkpointhandler.load(self.checkpoint_file, device, models_list, self.model_names, optimizers,
                                                 self.optimizer_names, schedulers, self.scheduler_names)

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
    def process_epochs(self, models_list, optimizers, schedulers,
                       train_loader, val_loader, test_loader):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self.load_checkpoint(models_list, optimizers, schedulers)

            # submit old stats to wandb to align with other runs
            self.report_previous_stats()

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            logging.info(f'stats:{self.stats}')
            # record start time
            start = time.time()

            # train
            train_stats = self.training.run_epoch(epoch, self.num_epochs, models_list, optimizers, train_loader)

            # validate
            val_stats = self.validation.run_epoch(epoch, self.num_epochs, models_list, val_loader)

            # testing
            # test_stats = self.testing.run_epoch(epoch, self.num_epochs, models_list, test_loader)
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

            # append current stats to all stats for checkpointing
            self.append_stats(curr_stats)

            # update scheduler for each model
            for i, model in enumerate(models_list):
                self.schedulerhandler.update_scheduler(schedulers[i], curr_stats)

            # save model checkpoint
            # if epoch % 5 == 0:
            self.checkpointhandler.save(models_list, self.model_names, optimizers, self.optimizer_names, schedulers,
                                        self.scheduler_names,
                                        epoch + 1, self.stats)

            # # check if early stopping criteria is met
            # if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
            #     logging.info('Early stopping criteria is met. Stopping the training process...')
            #     break  # stop running epochs
