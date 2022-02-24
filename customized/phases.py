import logging

import numpy as np
import torch
import torch.nn as nn


class PhaseHandler:

    def __init__(self):
        pass

    def get_train_phase(self, devicehandler, train_loader, wandb_config):
        training_phase = Training(devicehandler, train_loader, wandb_config)
        logging.info(f'Criterion for training phase:' +
                     f'\ngenerator:{training_phase.sn_criterion}\ndiscriminator:{training_phase.en_criterion}')
        return training_phase

    def get_val_phase(self, devicehandler, val_loader, wandb_config):
        validation_phase = Validation(devicehandler, val_loader, wandb_config)
        logging.info(f'Criterion for validation phase:\ngenerator:{validation_phase.criterion}')
        return validation_phase

    def get_test_phase(self, devicehandler, test_loader, wandb_config, output_dir):
        testing_phase = Testing(wandb_config, devicehandler, output_dir, test_loader)
        return testing_phase


def get_criterion(criterion_type):
    criterion = None
    if criterion_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'BCELoss':
        criterion = nn.BCELoss()
    return criterion


def calculate_num_hits(i, targets, out):
    # convert out to class labels
    labels_out = out.argmax(axis=1)
    if i == 0:
        logging.info(f'labels_out.shape:{labels_out.shape}')

    # count the total number of matches between predictions and actual
    compare = targets == labels_out
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


def _d_loss(pred, criterion, annotated=True):
    n = len(pred)

    if annotated:
        targets = torch.ones(n)  # targets should be 1.0
    else:
        targets = torch.zeros(n)  # targets should be 0.0

    loss = criterion(pred.squeeze(-1).cpu(), targets)  # make same dimensions

    return loss


class Training:
    def __init__(self, devicehandler, dataloader, wandb_config):
        self.devicehandler = devicehandler
        self.dataloader = dataloader

        self.sn_criterion = get_criterion(wandb_config.sn_criterion)
        self.en_criterion = get_criterion(wandb_config.en_criterion)
        self.use_gan = wandb_config.use_gan
        self.sigma = wandb_config.sigma
        self.sigma_weight = wandb_config.sigma_weight
        self.gan_start_epoch = wandb_config.gan_start_epoch

    def run_epoch(self, epoch, num_epochs, models, optimizers):
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')

        total_g_train_loss = 0
        total_d_train_loss_unannotated = 0
        total_d_train_loss_annotated = 0
        total_d_train_loss = 0

        g_model = models[0]
        d_model = models[1]
        g_optimizer = optimizers[0]
        d_optimizer = optimizers[1]

        # Set model in 'Training mode'
        g_model.train()
        d_model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.dataloader):
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
                g_loss, d_loss_unannotated, d_loss_annotated, d_loss = losses

                # append losses to running totals
                total_d_train_loss_unannotated += d_loss_unannotated.item()
                total_d_train_loss_annotated += d_loss_annotated.item()
                total_d_train_loss += d_loss.item()

            # compute backward pass of generator
            g_loss.backward()

            # update generator weights
            g_optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

            # append losses to running totals
            total_g_train_loss += g_loss.item()

        # calculate average loss across all mini-batches
        total_g_train_loss /= len(self.dataloader)
        total_d_train_loss /= len(self.dataloader)
        total_d_train_loss_unannotated /= len(self.dataloader)
        total_d_train_loss_annotated /= len(self.dataloader)

        # build stat dictionary
        g_lr = g_optimizer.state_dict()["param_groups"][0]["lr"]
        d_lr = d_optimizer.state_dict()["param_groups"][0]["lr"]
        stats = {'g_train_loss': total_g_train_loss, 'd_train_loss': total_d_train_loss,
                 'd_train_loss_unannotated': total_d_train_loss_unannotated,
                 'd_train_loss_annotated': total_d_train_loss_annotated,
                 'g_lr': g_lr, 'd_lr': d_lr}

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
        d_loss_unannotated = _d_loss(unannotated_pred, self.en_criterion, annotated=False)

        # 2 - compute forward pass on discriminator using annotated data
        # combine inputs and probability map
        annotated_inputs = inputs[annotated_idx]  # (B, C, H, W)
        annotated_targets = targets[annotated_idx]  # (B, H, W) targets only has a single class
        d_input = _combine_input_and_map(annotated_inputs, annotated_targets.unsqueeze(1))  # unsqueeze to match inputs

        # forward pass
        annotated_pred = d_model(d_input.detach(), i)  # detach to not affect generator?

        # calculate loss
        d_loss_annotated = _d_loss(annotated_pred, self.en_criterion, annotated=True)

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
        total_g_loss = g_loss + sigma * _d_loss(fake_pred, self.en_criterion, annotated=True)

        return total_g_loss, d_loss_unannotated, d_loss_annotated, d_loss


class Validation:
    def __init__(self, devicehandler, dataloader, wandb_config):
        self.devicehandler = devicehandler
        self.dataloader = dataloader
        self.criterion = get_criterion(wandb_config.sn_criterion)

    def run_epoch(self, epoch, num_epochs, models):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        total_val_loss = 0
        total_hits = 0
        total_iou_score = 0
        out_shape = None  # save for calculating total number of pixels per image

        g_model = models[0]
        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            g_model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.dataloader):
                logging.info(f'validation batch:{i}')

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(g_model, inputs, targets)

                # compute forward pass
                out = g_model.forward(inputs, i)
                out_shape = out.shape

                if i == 0:
                    logging.info(f'inputs.shape:{inputs.shape}')
                    logging.info(f'targets.shape:{targets.shape}')
                    logging.info(f'out.shape:{out.shape}')

                # calculate loss
                loss = self.criterion(out, targets)
                total_val_loss += loss.item()

                # calculate accuracy
                total_hits += calculate_num_hits(i, targets, out)
                total_iou_score += calculate_iou_score(i, targets, out)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate average evaluation metrics per mini-batch
            pixels_per_image = out_shape[2] * out_shape[3]
            possible_hits = len(self.dataloader.dataset) * pixels_per_image
            val_acc = total_hits / possible_hits
            total_val_loss /= len(self.dataloader)
            total_iou_score /= len(self.dataloader.dataset)

            # build stats dictionary
            stats = {'val_loss': total_val_loss, 'val_acc': val_acc, 'val_iou_score': total_iou_score}

            return stats


# TODO: format and save output
class Testing:
    def __init__(self, wandb_config, devicehandler, outputhandler, test_loader):
        self.devicehandler = devicehandler
        self.outputhandler = outputhandler
        self.test_loader = test_loader

    def run_epoch(self, epoch, num_epochs, models):
        logging.info(f'Running epoch {epoch}/{num_epochs} of testing...')

        # g_model = models[0]
        # with torch.no_grad():  # deactivate autograd engine to improve efficiency
        #
        #     # Set model in validation mode
        #     g_model.eval()
        #
        #     # process mini-batches
        #     for i, (inputs, targets) in enumerate(self.test_loader):
        #         # prep
        #         inputs, targets = self.devicehandler.move_data_to_device(g_model, inputs, targets)
        #
        #         # compute forward pass
        #         out = g_model.forward(inputs)
        #
        #         # format and save output

        return {}  # empty dictionary
