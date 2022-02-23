import logging

import numpy as np
import torch
import torch.nn as nn


def get_phases(wandb_config, devicehandler, outputhandler, train_loader, val_loader, test_loader):
    training_phase = Training(wandb_config, devicehandler, train_loader)
    validation_phase = Validation(wandb_config, devicehandler, val_loader)
    testing_phase = Testing(wandb_config, devicehandler, outputhandler, test_loader)
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


def _combine_input_and_map(input, map):
    combined = torch.cat((input, map), dim=1)
    return combined


def _d_loss(pred, annotated=True):
    criterion = nn.BCELoss()

    n = len(pred)

    if annotated:
        targets = torch.ones(n)  # targets should be 1.0
    else:
        targets = torch.zeros(n)  # targets should be 0.0

    loss = criterion(pred.squeeze(-1).cpu(), targets)  # make same dimensions

    return loss


# TODO: add in d model component / gan
class Training:
    def __init__(self, wandb_config, devicehandler, train_loader):
        self.devicehandler = devicehandler
        self.sn_criterion = get_criterion(wandb_config.sn_criterion)
        self.en_criterion = get_criterion(wandb_config.sn_criterion)
        self.use_gan = wandb_config.use_gan
        self.train_loader = train_loader
        self.sigma = wandb_config.sigma
        self.sigma_weight = wandb_config.sigma_weight
        self.gan_start_epoch = wandb_config.gan_start_epoch

        logging.info(f'Generator criterion for training phase:\n{self.sn_criterion}')
        logging.info(f'Discriminator criterion for training phase:\n{self.en_criterion}')

    def run_epoch(self, epoch, num_epochs, models, optimizers):
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')

        g_train_loss = 0
        d_train_loss_unannotated = 0
        d_train_loss_annotated = 0
        d_train_loss = 0

        g_model = models[0]
        d_model = models[1]
        g_optimizer = optimizers[0]
        d_optimizer = optimizers[1]

        # Set model in 'Training mode'
        g_model.train()
        d_model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
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

            # calculate generator loss
            g_loss = self.sn_criterion(out, targets)

            # check if gan process should be run
            if self.use_gan and epoch >= self.gan_start_epoch:

                # run gan process
                losses = self.run_gan(i, epoch, inputs, out, targets, d_model, d_optimizer, g_loss)

                # unpack losses
                total_g_loss, d_loss_unannotated, d_loss_annotated, d_loss = losses

                # append losses to running totals
                d_train_loss_unannotated += d_loss_unannotated.item()
                d_train_loss_annotated += d_loss_annotated.item()
                d_train_loss += d_loss.item()
            else:
                total_g_loss = g_loss

            # compute backward pass of generator
            total_g_loss.backward()

            # update generator weights
            g_optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

            # append losses to running totals
            g_train_loss += total_g_loss.item()

        # calculate average loss across all mini-batches
        g_train_loss /= len(self.train_loader)
        d_train_loss /= len(self.train_loader)

        # build stat dictionary
        g_lr = g_optimizer.state_dict()["param_groups"][0]["lr"]
        stats = {'g_train_loss': g_train_loss, 'd_train_loss': d_train_loss,
                 'd_train_loss_unannotated': d_train_loss_unannotated, 'd_train_loss_annotated': d_train_loss_annotated,
                 'g_lr': g_lr}

        return stats

    def run_gan(self, i, epoch, inputs, out, targets, d_model, d_optimizer, g_loss):

        # select subset of mini-batch to be unannotated vs annotated at random
        unannotated_idx = np.random.choice(len(inputs), size=int(len(inputs) / 2), replace=False)
        annotated_idx = np.delete(np.array([k for k in range(len(inputs))]), unannotated_idx)

        # 1 - compute forward pass on discriminator using unannotated data
        # combine inputs and probability map
        unannotated_inputs = inputs[unannotated_idx]  # (B, C, H, W)
        unannotated_out = out[unannotated_idx, 0, :, :]  # keep 1 class to match inputs + targets shape, get (B, H, W)
        d_input = _combine_input_and_map(unannotated_inputs, unannotated_out.unsqueeze(1))  # unsqueeze to match inputs

        # forward pass
        unannotated_pred = d_model(d_input.detach(), i)  # detach to not affect generator?

        # calculate loss
        d_loss_unannotated = _d_loss(unannotated_pred, annotated=False)

        # 2 - compute forward pass on discriminator using annotated data
        # combine inputs and probability map
        annotated_inputs = inputs[annotated_idx]  # (B, C, H, W)
        annotated_targets = targets[annotated_idx]  # (B, H, W) targets only has a single class
        d_input = _combine_input_and_map(annotated_inputs, annotated_targets.unsqueeze(1))  # unsqueeze to match inputs

        # forward pass
        annotated_pred = d_model(d_input.detach(), i)  # detach to not affect generator?

        # calculate loss
        d_loss_annotated = _d_loss(annotated_pred, annotated=True)

        # 3 - update discriminator based on loss
        # calculate total discriminator loss for unannotated and annotated data
        sigma = self.sigma
        sigma += (epoch / self.sigma_weight)  # add more weight each time
        d_loss = sigma * (d_loss_unannotated + d_loss_annotated)  # Should we consider sigma here or only for generator?
        d_loss.backward()
        d_optimizer.step()

        # 4 - compute forward pass on updated discriminator using only unannotated data for calculating generator loss
        # combine inputs and probability map
        # can I use all output here or only the ones selected for unannotation?
        unannotated_out = out[:, 0, :, :]  # keep 1 class to match inputs + targets shape, get (B, H, W)
        d_input = _combine_input_and_map(inputs, unannotated_out.unsqueeze(1))  # unsqueeze to match inputs

        # forward pass
        fake_pred = d_model(d_input, i)  # leave attached so backpropagation through discriminator affects generator

        # calculate generator loss based on discriminator predictions
        # if discriminator predicts unannotated correctly, generator not doing good enough job
        total_g_loss = g_loss + sigma * _d_loss(fake_pred, annotated=True)

        return total_g_loss, d_loss_unannotated, d_loss_annotated, d_loss


class Validation:
    def __init__(self, wandb_config, devicehandler, val_loader):
        self.devicehandler = devicehandler
        self.criterion = get_criterion(wandb_config['sn_criterion'])
        self.resize_height = wandb_config.resize_height
        self.resize_width = wandb_config.resize_width
        self.val_loader = val_loader

        logging.info(f'Generator criterion for validation phase:\n{self.criterion}')

    def run_epoch(self, epoch, num_epochs, models):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        val_loss = 0
        actual_hits = 0
        score = 0

        sn_model = models[0]
        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set SN model in validation mode
            sn_model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
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
            possible_hits = len(self.val_loader.dataset) * self.resize_width * self.resize_height
            val_loss /= len(self.val_loader)
            val_acc = actual_hits / possible_hits
            iou_score = score / len(self.val_loader.dataset)

            # build stats dictionary
            stats = {'val_loss': val_loss, 'val_acc': val_acc, 'val_iou_score': iou_score}

            return stats


# TODO: format and save output
class Testing:
    def __init__(self, wandb_config, devicehandler, outputhandler, test_loader):
        self.devicehandler = devicehandler
        self.outputhandler = outputhandler
        self.test_loader = test_loader

    def run_epoch(self, epoch, num_epochs, models):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        sn_model = models[0]
        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            sn_model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.test_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(sn_model, inputs, targets)

                # compute forward pass
                out = sn_model.forward(inputs)

                # format and save output

            return {}  # empty dictionary
