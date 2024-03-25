import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import numpy as np

from utils.gleason_torch_utils import ProstateDataset
from utils.data_ki67 import Ki67Dataset

def build_data(dataset_name="Ki67", batch_size=128):
    if dataset_name == "MNIST":
        """mnist"""
        data_train = MNIST("../data/",
                        download=True,
                        train=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

        data_val = MNIST("../data/",
                        train=False,
                        download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

    elif dataset_name == "prostate":
        """define gleason_CNN data generators"""
        data_train = ProstateDataset(state="train")
        data_val = ProstateDataset(state="val")
    elif dataset_name == "Ki67":
        data_train = Ki67Dataset(state="train")
        data_val =Ki67Dataset(state="val")
        data_test =Ki67Dataset(state="test")
    else:
        print(f"Please check your dataset: {dataset_name} again!")


    dataloader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(data_val, batch_size=batch_size, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, num_workers=8)

    dataloaders = {
        "train": dataloader_train,
        "val": dataloader_val,
        "test": dataloader_test
    }

    # digit_one, _ = data_val[5]

    # return dataloaders, digit_one
    return dataloaders, None
