import logging
import time


# TODO: move stats into its own handler
# TODO: report old stats during reload
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

    def process_epochs(self, config, models, model_names, optimizers, optimizer_names, schedulers, scheduler_names,
                       training,
                       evaluation,
                       testing):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self.load_checkpoint(models, model_names, optimizers, optimizer_names, schedulers, scheduler_names)

            # # submit old stats to wandb to align with other runs
            # self.statshandler.report_previous_stats(self.wandbconnector)

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_stats = training.train_model(epoch, self.num_epochs, models, optimizers, config)

            # validate
            val_stats = evaluation.evaluate_model(epoch, self.num_epochs, models, config)

            # testing
            test_stats = testing.test_model(epoch, self.num_epochs, models)

            # report stats
            end = time.time()
            # lr = g_optimizer.state_dict()["param_groups"][0]["lr"]
            # self.statshandler.collect_stats(epoch, lr, g_train_loss, d_train_loss, d_train_loss_unannotated,
            #                                 d_train_loss_annotated, val_loss, val_metric, iou_score,
            #                                 start, end)
            # self.statshandler.report_stats(self.wandbconnector)

            # update scheduler for each model
            for i, model in enumerate(models):
                self.schedulerhandler.update_scheduler(schedulers[i], self.stats)

            # save model checkpoint
            if epoch % 5 == 0:
                self.checkpointhandler.save(models, model_names, optimizers, optimizer_names, schedulers, scheduler_names,
                                            epoch + 1, self.stats)

            # # check if early stopping criteria is met
            # if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
            #     logging.info('Early stopping criteria is met. Stopping the training process...')
            #     break  # stop running epochs
