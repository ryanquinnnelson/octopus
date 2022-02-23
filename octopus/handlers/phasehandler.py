import logging
import time


# TODO: check against early stopping criteria
class PhaseHandler:
    def __init__(self, num_epochs, devicehandler, checkpointhandler, schedulerhandler, wandbconnector,
                 load_from_checkpoint,
                 checkpoint_file=None):
        logging.info(f'Initializing phase handler...')

        self.devicehandler = devicehandler
        self.checkpointhandler = checkpointhandler
        self.wandbconnector = wandbconnector
        self.schedulerhandler = schedulerhandler

        self.checkpoint_file = checkpoint_file
        self.load_from_checkpoint = load_from_checkpoint
        self.stats = {}
        self.first_epoch = 1
        self.num_epochs = num_epochs

    def load_checkpoint(self, models, model_names, optimizers, optimizer_names, schedulers, scheduler_names):
        device = self.devicehandler.get_device()
        checkpoint = self.checkpointhandler.load(self.checkpoint_file, device, models, model_names, optimizers,
                                                 optimizer_names, schedulers, scheduler_names)

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

    def process_epochs(self, wandb_config, models, model_names, optimizers, optimizer_names, schedulers,
                       scheduler_names,
                       training,
                       evaluation,
                       testing):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self.load_checkpoint(models, model_names, optimizers, optimizer_names, schedulers, scheduler_names)

            # submit old stats to wandb to align with other runs
            self.report_previous_stats()

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_stats = training.run_epoch(epoch, self.num_epochs, models, optimizers, wandb_config)

            # validate
            val_stats = evaluation.run_epoch(epoch, self.num_epochs, models, wandb_config)

            # testing
            test_stats = testing.run_epoch(epoch, self.num_epochs, models, wandb_config)

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
            for i, model in enumerate(models):
                self.schedulerhandler.update_scheduler(schedulers[i], curr_stats)

            # save model checkpoint
            if epoch % 5 == 0:
                self.checkpointhandler.save(models, model_names, optimizers, optimizer_names, schedulers,
                                            scheduler_names,
                                            epoch + 1, self.stats)

            # # check if early stopping criteria is met
            # if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
            #     logging.info('Early stopping criteria is met. Stopping the training process...')
            #     break  # stop running epochs
