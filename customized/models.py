import logging
from collections import OrderedDict

import torch
import torch.nn as nn


# TODO: revise models to modular
def get_models(wandb_config):
    sn = SegmentationNetwork()
    en = SegmentationNetwork()

    logging.info(f'Model1 initialized:\n{sn}')
    logging.info(f'Model2 initialized:\n{en}')

    return [sn, en]


def _init_weights(layer):
    """
    Perform initialization of layer weights if layer is a Conv2d layer.
    Args:
        layer: layer under consideration
    Returns: None
    """
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)


class CnnBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # self.input_size = input_size
        # self.output_size = _calc_output_size(input_size, padding, dilation, kernel_size, stride)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # initialize weights
        self.cnn_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'cnn_block_input:{x.shape}')
        x = self.cnn_block(x)
        # logging.info(f'cnn_block:{x.shape}')
        return x


# bi-linear interpolation, or learned up-sampling filters
# nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear')
# https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, size, mode='bilinear'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_block = nn.Sequential(
            nn.Upsample(size=size, mode=mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # initialize weights
        self.up_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'up_block_input:{x.shape}')
        x = self.up_block(x)
        # logging.info(f'up_block:{x.shape}')
        return x


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, batchnorm=True, activation='relu'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear_block = nn.Sequential()
        self.linear_block.add_module('linear', nn.Linear(in_features, out_features))

        if batchnorm:
            self.linear_block.add_module('bn', nn.BatchNorm1d(out_features))

        if activation == 'relu':
            self.linear_block.add_module('activation', nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            self.linear_block.add_module('activation', nn.Sigmoid())

        self.linear_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'linear_block_input:{x.shape}')
        x = self.linear_block(x)
        # logging.info(f'linear_block:{x.shape}')
        return x


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            # conv1
            CnnBlock(3, 64),
            CnnBlock(64, 64),

            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv2
            CnnBlock(64, 128),

            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv3
            CnnBlock(128, 128),
            CnnBlock(128, 256),

            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv4
            CnnBlock(256, 512),
            CnnBlock(512, 512)  # shortcut to up-conv1

        )

        self.block2 = nn.Sequential(

            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv5
            CnnBlock(512, 512),
            CnnBlock(512, 512)  # shortcut to up-conv2

        )

        self.block3 = nn.Sequential(

            # pool5
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv6
            CnnBlock(512, 1024),
            CnnBlock(1024, 1024)  # shortcut to up-conv3
        )

        self.block4 = UpConvBlock(1024, 1024, (224, 332))

        self.block5 = UpConvBlock(512, 512, (224, 332))

        self.block6 = UpConvBlock(512, 512, (224, 332))

        self.block7 = nn.Sequential(
            CnnBlock(2048, 1024),
            nn.Conv2d(1024, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):
        block1out = self.block1(x)
        block2out = self.block2(block1out)
        block3out = self.block3(block2out)

        # upconvolution
        block4out = self.block4(block3out)
        block5out = self.block5(block2out)
        block6out = self.block6(block1out)

        # concatenate results
        concatenated = torch.cat((block4out, block5out, block6out), dim=1)  # channels are the second dimension

        block7out = self.block7(concatenated)

        if i == 0:
            logging.info(f'block1out.shape:{block1out.shape}')
            logging.info(f'block2out.shape:{block2out.shape}')
            logging.info(f'block3out.shape:{block3out.shape}')
            logging.info(f'block4out.shape:{block4out.shape}')
            logging.info(f'block5out.shape:{block5out.shape}')
            logging.info(f'block6out.shape:{block6out.shape}')
            logging.info(f'concatenated.shape:{concatenated.shape}')
            logging.info(f'block7out.shape:{block7out.shape}')

        return block7out
