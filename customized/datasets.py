"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import customized.helper as helper


class DatasetHandler:
    def __init__(self, data_dir):
        """
        Initialize NumericalDatasetHandler.
        :param data_dir (str): fully qualified path to root directory inside which data subdirectories are placed
        :param train_data (str): fully qualified path to training data
        :param val_data (str): fully qualified path to validation data
        :param train_class (Dataset): torch Dataset class to use for training data
        :param val_class (Dataset): torch Dataset class to use for validation data
        """

        self.data_dir = data_dir

    def get_train_dataset(self, config):
        """
        Load training data into memory and initialize the Dataset object.
        :return: Dataset
        """
        # parse configs
        train_dir = config['data']['train_dir']
        train_target_dir = config['data']['train_target_dir']
        train_transforms = helper.to_string_list(config['data']['transforms_list'])
        resize_height = config['data'].getint('resize_height')

        # initialize dataset
        t = _compose_transforms(train_transforms, resize_height)
        dataset = ImageDataset(train_dir, train_target_dir, t)
        logging.info(f'Loaded {len(dataset)} training images.')
        return dataset

    def get_val_dataset(self, config):
        """
        Load validation data into memory and initialize the Dataset object.
        :return: Dataset
        """
        # parse configs
        val_dir = config['data']['val_dir']
        val_target_dir = config['data']['val_target_dir']
        train_transforms = helper.to_string_list(config['data']['transforms_list'])
        resize_height = config['data'].getint('resize_height')

        # determine whether normalize transform should also be applied to validation and test data
        self.should_normalize_val = True if 'Normalize' in train_transforms else False

        if self.should_normalize_val:
            logging.info('Normalizing validation data to match normalization of training data...')
            t = _compose_transforms(['Resize', 'ToTensor', 'Normalize'], resize_height)
        else:
            t = _compose_transforms(['Resize', 'ToTensor'], resize_height)

        # initialize dataset
        dataset = ImageDataset(val_dir, val_target_dir, t)
        logging.info(f'Loaded {len(dataset)} validation images.')
        return dataset

    def get_test_dataset(self, config):
        """
        Load validation data into memory and initialize the Dataset object.
        :return: Dataset
        """
        # parse configs
        test_dir = config['data']['test_dir']
        test_target_dir = config['data']['test_target_dir']
        train_transforms = helper.to_string_list(config['data']['transforms_list'])
        resize_height = config['data'].getint('resize_height')

        # determine whether normalize transform should also be applied to validation and test data
        self.should_normalize_test = True if 'Normalize' in train_transforms else False
        if self.should_normalize_test:
            logging.info('Normalizing validation data to match normalization of training data...')
            t = _compose_transforms(['Resize', 'ToTensor', 'Normalize'], resize_height)
        else:
            t = _compose_transforms(['Resize', 'ToTensor'], resize_height)

        # initialize dataset
        dataset = ImageDataset(test_dir, test_target_dir, t)
        logging.info(f'Loaded {len(dataset)} test images.')
        return dataset


def _apply_transformations(img, target):
    if random.random() > 0.5:
        # print('vflip')
        img = transforms.functional_pil.vflip(img)
        target = transforms.functional_pil.vflip(target)

    if random.random() > 0.5:
        # print('hflip')
        img = transforms.functional_pil.hflip(img)
        target = transforms.functional_pil.hflip(target)

    return img, target


class ImageDataset(Dataset):

    def __init__(self, img_dir, targets_dir, transform=None):
        self.img_dir = img_dir
        self.targets_dir = targets_dir
        self.transform = transform

        # prepare image list
        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list.remove('.DS_Store')  # remove mac generated files
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])

        # generate target name
        # target image name matches image but also includes suffix
        img_name = self.img_list[idx][:-4]  # strip .bmp
        target_path = os.path.join(self.targets_dir, img_name + '_anno.bmp')

        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        # standardize image size based on original image size
        img = img.resize((775, 522), resample=Image.BILINEAR)  # standardize image size
        target = target.resize((775, 522), resample=Image.BILINEAR)  # standardize target size

        # apply matching transformations to image and target
        img, target = _apply_transformations(img, target)

        # transform and convert to tensors
        tensor_img = self.transform(img)
        tensor_target = self.transform(target)

        # keep only first channel because all three channels are given the same value
        tensor_target_first_channel = tensor_target[0]

        # convert all nonzero target values to 1
        # nonzero values indicate segment
        # zero values indicate background
        tensor_target_first_channel[tensor_target_first_channel != 0] = 1.0

        # convert target to long datatype to indicate classes
        tensor_target_first_channel = tensor_target_first_channel.to(torch.long)

        return tensor_img, tensor_target_first_channel


def _compose_transforms(transforms_list, resize_height):
    """
    Build a composition of transformations to perform on image data.
    Args:
        transforms_list (List): list of strings representing individual transformations,
        in the order they should be performed
    Returns: transforms.Compose object containing all desired transformations
    """
    t_list = []

    for each in transforms_list:
        if each == 'RandomHorizontalFlip':
            t_list.append(transforms.RandomHorizontalFlip(0.1))  # customized because 0.5 is too much
        elif each == 'ToTensor':
            t_list.append(transforms.ToTensor())
        elif each == 'Resize':
            # t_list.append(transforms.Resize((775, 522), interpolation='bilinear'))
            t_list.append(transforms.Resize(resize_height, interpolation=Image.BILINEAR))

    composition = transforms.Compose(t_list)

    return composition
