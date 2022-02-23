import logging

import numpy as np
import torch


def get_training_phase():
    return None


def get_validation_phase():
    return None


def get_testing_phase():
    return None


def get_phases(wandb_config):
    return None, None, None

#
# def run_training_epoch(epoch, num_epochs, models, optimizers, wandb_config):
#     return None
#
#
# def _calculate_num_hits(i, targets, out):
#     # convert to class labels
#     # convert out to class labels
#     labels_out = out.argmax(axis=1)
#     if i == 0:
#         logging.info(f'labels_out.shape:{labels_out.shape}')
#
#     # compare predictions against actual
#     compare = targets == labels_out
#
#     # # convert 2D images into 1D vectors
#     # out = labels_out.cpu().detach().numpy().reshape((batch_size, -1))
#     # labels_inputs = inputs.cpu().detach().numpy().reshape((batch_size, -1))
#
#     # compare lists of max indices and find the number that match
#     n_hits = np.sum(compare.cpu().detach().numpy())
#
#     if i == 0:
#         logging.info(f'n_hits:{n_hits}')
#
#     return n_hits
#
#
# # https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
# def _calculate_iou_score(i, targets, out):
#     targets = targets.cpu().detach().numpy()
#
#     # convert to class labels
#     # convert out to class labels
#     labels_out = out.argmax(axis=1)
#     labels_out = labels_out.cpu().detach().numpy()
#
#     intersection = np.logical_and(targets, labels_out)
#     union = np.logical_or(targets, labels_out)
#
#     iou_score = np.sum(intersection) / np.sum(union)
#     return iou_score
#
# def run_validation_epoch(epoch, num_epochs, models, wandb_config):
#     logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')
#
#     val_loss = 0
#     actual_hits = 0
#     score = 0
#
#     with torch.no_grad():  # deactivate autograd engine to improve efficiency
#
#         # Set model in validation mode
#         g_model.eval()
#
#         # process mini-batches
#         for i, (inputs, targets) in enumerate(self.val_loader):
#             logging.info(f'validation batch:{i}')
#
#             # prep
#             inputs, targets = self.devicehandler.move_data_to_device(g_model, inputs, targets)
#
#             # compute forward pass
#             out = g_model.forward(inputs, i)
#
#             if i == 0:
#                 logging.info(f'inputs.shape:{inputs.shape}')
#                 logging.info(f'targets.shape:{targets.shape}')
#                 logging.info(f'out.shape:{out.shape}')
#
#             # calculate loss
#             loss = self.criterion(out, targets)
#             val_loss += loss.item()
#
#             # calculate accuracy
#             actual_hits += _calculate_num_hits(i, targets, out)
#             score += _calculate_iou_score(i, targets, out)
#
#             # delete mini-batch from device
#             del inputs
#             del targets
#
#         # calculate evaluation metrics
#         possible_hits = (len(self.val_loader.dataset) * 224 * 332)
#         val_loss /= len(self.val_loader)  # average per mini-batch
#         val_acc = actual_hits / possible_hits
#         iou_score = score / len(self.val_loader.dataset)
#
#         return val_loss, val_acc, iou_score
#
#
# def run_test_epoch(epoch, num_epochs, models, wandb_config):
#     return None
