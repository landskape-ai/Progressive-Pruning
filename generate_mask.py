import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from advertorch.utils import NormalizeByChannelMeanStd
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from models.ResNets import resnet20s
from tools.pruning_utils import *
from utils import setup_model


def generate_mask_(args, data, pruner, model_dir, save, state, gpu=0):
    def prune_loop(
        model,
        loss,
        pruner,
        dataloader,
        device,
        sparsity,
        scope,
        epochs,
        train_mode=False,
    ):

        # Set model to train or eval mode
        model.train()
        if not train_mode:
            model.eval()

        # Prune model
        for epoch in range(epochs):
            pruner.score(model, loss, dataloader, device)
            sparse = sparsity ** ((epoch + 1) / epochs)
            pruner.mask(sparse, scope)

    torch.cuda.set_device(int(gpu))

    model = setup_model(args)
    prune_conv(model)

    print("loading model from {}".format(model_dir))
    checkpoint = torch.load(model_dir, map_location="cpu")
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=False)

    model.cuda()

    remain_weight = 0.8 ** state

    if pruner == "mag":
        print("Pruning with Magnitude")
        pruner = Mag(masked_parameters(model))
        prune_loop(
            model,
            None,
            pruner,
            None,
            torch.device("cuda:{}".format(gpu)),
            remain_weight,
            scope=args.scope,
            epochs=10,
            train_mode=True,
        )
        current_mask = extract_mask(model.state_dict())
        check_sparsity_dict(current_mask)
        torch.save(current_mask, save)

    elif pruner == "snip":
        print("Pruning with SNIP")
        criterion = nn.CrossEntropyLoss()

        data_loader = DataLoader(
            data, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
        )

        pruner = SNIP(masked_parameters(model))
        prune_loop(
            model,
            criterion,
            pruner,
            data_loader,
            torch.device("cuda:{}".format(gpu)),
            remain_weight,
            scope=args.scope,
            epochs=1,
            train_mode=True,
        )
        current_mask = extract_mask(model.state_dict())
        check_sparsity_dict(current_mask)
        torch.save(current_mask, save)

    elif pruner == "random":
        print("Pruning with Magnitude")
        pruner = Rand(masked_parameters(model))
        prune_loop(
            model,
            None,
            pruner,
            None,
            torch.device("cuda:{}".format(gpu)),
            remain_weight,
            scope=args.scope,
            epochs=1,
            train_mode=True,
        )
        current_mask = extract_mask(model.state_dict())
        check_sparsity_dict(current_mask)
        torch.save(current_mask, save)

    elif pruner == "GraSP":
        print("Pruning with GraSP")
        criterion = nn.CrossEntropyLoss()
     
        trainset = torchvision.datasets.CIFAR10(
            args.data, train=True, download=True, transform=transforms.ToTensor()
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2
        )

        pruner = GraSP(masked_parameters(model))
        prune_loop(
            model,
            criterion,
            pruner,
            trainloader,
            torch.device("cuda:{}".format(gpu)),
            remain_weight,
            scope="global",
            epochs=1,
            train_mode=True,
        )
        current_mask = extract_mask(model.state_dict())
        check_sparsity_dict(current_mask)
        torch.save(current_mask, save)
