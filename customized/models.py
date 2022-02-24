import logging

import torch.nn as nn


class ModelHandler:
    def __init__(self):
        pass

    # TODO: revise models to modular, accept argument for which model type to use
    def get_models(self, wandb_config):
        sn = SegmentationNetwork()
        en = EvaluationNetwork()

        logging.info(f'Generator model initialized:\n{sn}')
        logging.info(f'Discriminator model initialized:\n{en}')

        return [sn, en], ['sn_model', 'en_model']


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

        self.block7 = nn.Sequential(
            CnnBlock(3, 3),
            nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):
        block7out = self.block7(x)

        if i == 0:
            logging.info(f'block7out.shape:{block7out.shape}')

        return block7out


class EvaluationNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(

            # conv1
            CnnBlock(4, 4)

        )

        self.block2 = nn.Sequential(

            nn.Flatten(),  # need to convert 2d to 1d

        )

        self.block3 = nn.Sequential(
            LinearBlock(297472, 256),  # 4*224*332
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            nn.Linear(64, 1),  # binary classes
            nn.Sigmoid()
        )

    def forward(self, x, i):
        if i == 0:
            logging.info(f'x:{x.shape}')

        block1out = self.block1(x)

        if i == 0:
            logging.info(f'block1out:{block1out.shape}')
        block2out = self.block2(block1out)

        if i == 0:
            logging.info(f'block2out:{block2out.shape}')

        block3out = self.block3(block2out)

        if i == 0:
            logging.info(f'block3out:{block3out.shape}')

        return block3out
