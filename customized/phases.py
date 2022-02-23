import logging

import numpy as np
import torch
import torch.nn as nn


def get_phases(wandb_config, devicehandler, outputhandler):
    training_phase = Training(wandb_config, devicehandler)
    validation_phase = Validation(wandb_config, devicehandler)
    testing_phase = Testing(wandb_config, devicehandler, outputhandler)
    return training_phase, validation_phase, testing_phase


def get_criterion(criterion_type):
    criterion = None
    if criterion_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    return criterion


def calculate_num_hits(i, targets, out):
    # convert to class labels
    # convert out to class labels
    labels_out = out.argmax(axis=1)
    if i == 0:
        logging.info(f'labels_out.shape:{labels_out.shape}')

    # compare predictions against actual
    compare = targets == labels_out

    # # convert 2D images into 1D vectors
    # out = labels_out.cpu().detach().numpy().reshape((batch_size, -1))
    # labels_inputs = inputs.cpu().detach().numpy().reshape((batch_size, -1))

    # compare lists of max indices and find the number that match
    n_hits = np.sum(compare.cpu().detach().numpy())

    if i == 0:
        logging.info(f'n_hits:{n_hits}')

    return n_hits


# https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def calculate_iou_score(i, targets, out):
    targets = targets.cpu().detach().numpy()

    # convert to class labels
    # convert out to class labels
    labels_out = out.argmax(axis=1)
    labels_out = labels_out.cpu().detach().numpy()

    intersection = np.logical_and(targets, labels_out)
    union = np.logical_or(targets, labels_out)

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# TODO: add in d model component / gan
class Training:
    def __init__(self, wandb_config, devicehandler):
        self.devicehandler = devicehandler
        self.sn_criterion = get_criterion(wandb_config.sn_criterion)
        self.en_criterion = get_criterion(wandb_config.sn_criterion)
        self.use_gan = wandb_config.use_gan

        logging.info(f'Generator criterion for training phase:\n{self.sn_criterion}')
        logging.info(f'Discriminator criterion for training phase:\n{self.en_criterion}')

    def run_epoch(self, epoch, num_epochs, models, optimizers, train_loader):
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')

        g_train_loss = 0
        d_train_loss_unannotated = 0
        d_train_loss_annotated = 0
        d_train_loss = 0

        g_model = models[0]
        # d_model = models[1]
        g_optimizer = optimizers[0]
        # d_optimizer = optimizers[1]

        # Set model in 'Training mode'
        g_model.train()
        # d_model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(train_loader):
            logging.info(f'training batch:{i}')
            # prep
            g_optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, targets = self.devicehandler.move_data_to_device(g_model, inputs, targets)

            # compute forward pass on generator
            out = g_model.forward(inputs, i)

            if i == 0:
                logging.info(f'inputs.shape:{inputs.shape}')
                logging.info(f'targets.shape:{targets.shape}')
                logging.info(f'out.shape:{out.shape}')

            # calculate loss
            g_loss = self.sn_criterion(out, targets)
            g_train_loss += g_loss.item()

            total_g_loss = g_loss

            # compute backward pass of generator
            total_g_loss.backward()

            # update generator weights
            g_optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        g_train_loss /= len(train_loader)
        d_train_loss /= len(train_loader)

        # build stat dictionary
        g_lr = g_optimizer.state_dict()["param_groups"][0]["lr"]
        stats = {'g_train_loss': g_train_loss, 'd_train_loss': d_train_loss,
                 'd_train_loss_unannotated': d_train_loss_unannotated, 'd_train_loss_annotated': d_train_loss_annotated,
                 'g_lr': g_lr}

        return stats


class Validation:
    def __init__(self, wandb_config, devicehandler):
        self.devicehandler = devicehandler
        self.criterion = get_criterion(wandb_config['sn_criterion'])
        self.resize_height = wandb_config.resize_height
        self.resize_width = wandb_config.resize_width

        logging.info(f'Generator criterion for validation phase:\n{self.criterion}')

    def run_epoch(self, epoch, num_epochs, models, val_loader):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        val_loss = 0
        actual_hits = 0
        score = 0

        sn_model = models[0]
        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set SN model in validation mode
            sn_model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(val_loader):
                logging.info(f'validation batch:{i}')

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(sn_model, inputs, targets)

                # compute forward pass
                out = sn_model.forward(inputs, i)

                if i == 0:
                    logging.info(f'inputs.shape:{inputs.shape}')
                    logging.info(f'targets.shape:{targets.shape}')
                    logging.info(f'out.shape:{out.shape}')

                # calculate loss
                loss = self.criterion(out, targets)
                val_loss += loss.item()

                # calculate accuracy
                actual_hits += calculate_num_hits(i, targets, out)
                score += calculate_iou_score(i, targets, out)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics per mini-batch
            possible_hits = len(val_loader.dataset) * self.resize_width * self.resize_height
            val_loss /= len(val_loader)
            val_acc = actual_hits / possible_hits
            iou_score = score / len(val_loader.dataset)

            # build stats dictionary
            stats = {'val_loss': val_loss, 'val_acc': val_acc, 'val_iou_score': iou_score}

            return stats


# TODO: format and save output
class Testing:
    def __init__(self, wandb_config, devicehandler, outputhandler):
        self.devicehandler = devicehandler
        self.outputhandler = outputhandler

    def run_epoch(self, epoch, num_epochs, models, test_loader):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        sn_model = models[0]
        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            sn_model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(test_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(sn_model, inputs, targets)

                # compute forward pass
                out = sn_model.forward(inputs)

                # format and save output

            return {}  # empty dictionary
