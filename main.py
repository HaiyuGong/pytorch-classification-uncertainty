import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models

import numpy as np
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import yaml
import os

from helpers import get_device, rotate_img, one_hot_embedding
from utils.data import build_data
from train import train_model
from test import rotating_image_classification, test_single_image, test_Ki67
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet

def read_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config_file = "configs/base_ki67_train.yaml"
    cfg = read_config(config_file)


    """数据加载"""
    dataloaders, digit_one = build_data(cfg['dataset_name'], batch_size=cfg['batch_size'])
    print("Load dataset successfully!")

    num_epochs = cfg['epochs']
    use_uncertainty = cfg['uncertainty']
    num_classes = cfg['num_classes']

    # get device
    if torch.cuda.is_available():
        device = torch.device(cfg['device'])
    else:
        raise ValueError("GPU Device not available.")
    
    # if cfg['model'] == 'LeNet':
    # model = LeNet(dropout=cfg['dropout'])
    # elif cfg['model'] == 'resnet50':
    model = models.resnet50(num_classes=num_classes)
    model.load_state_dict(torch.load(r'results/Ki67/model.pt')['model_state_dict'])
    model = model.to(device)

    if cfg['mode']=='example':
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig("./images/examples.jpg")

    elif cfg['mode']=='train':

        os.makedirs(os.path.join(cfg['result_dir'], cfg['dataset_name']), exist_ok=True)
        if use_uncertainty:
            model_save_path=os.path.join(cfg['result_dir'], cfg['dataset_name'], 'model_uncertainty_{}_pretrain.pt'.format(cfg['criterion']))
            if cfg['criterion'] == 'digamma':
                criterion = edl_digamma_loss
            elif cfg['criterion'] == 'log':
                criterion = edl_log_loss
            elif cfg['criterion'] == 'mse':
                criterion = edl_mse_loss
            else:
                raise ValueError("'uncertainty=True' requires mse, log or digamma.")
        else:
            criterion = nn.CrossEntropyLoss()
            model_save_path=os.path.join(cfg['result_dir'], cfg['dataset_name'], 'model.pt')


        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  

        model, metrics = train_model(
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            cfg['log_dir'],
            scheduler=exp_lr_scheduler,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
        )

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, model_save_path)
        print(f"Saved: {model_save_path}")

    elif cfg['mode']=='test':

        if use_uncertainty:
            model_save_path=os.path.join(cfg['result_dir'], cfg['dataset_name'], 'model_uncertainty_{}_pretrain.pt'.format(cfg['criterion']))
            # filename = os.path.join(cfg['result_dir'], cfg['dataset_name'], 'rotate_uncertainty_{}.jpg'.format(cfg['criterion']))
        else:
            model_save_path=os.path.join(cfg['result_dir'], cfg['dataset_name'], 'model.pt')
            # checkpoint = torch.load("./results/model.pt")
            # filename = os.path.join(cfg['result_dir'], cfg['dataset_name'], 'rotate.jpg')
        checkpoint = torch.load(model_save_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.eval()

        test_Ki67(model,dataloaders,use_uncertainty,phase='test',device=device)

        # rotating_image_classification(
        #     model, digit_one, filename, uncertainty=use_uncertainty
        # )

        # test_single_image(model, "../data/MNIST/one.jpg", uncertainty=use_uncertainty)
        # test_single_image(model, "../data/MNIST/yoda.jpg", uncertainty=use_uncertainty)


if __name__ == "__main__":
    main()
