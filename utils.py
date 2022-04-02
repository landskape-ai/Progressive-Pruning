"""
    setup model and datasets
"""

from advertorch.utils import NormalizeByChannelMeanStd

import torch
import torch.nn as nn
# from advertorch.utils import NormalizeByChannelMeanStd
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from dataset import *
from models import *

__all__ = ["setup_model_dataset", "setup_model"]


def evaluate_cer(net, args,loader_=None):
    criterion = nn.CrossEntropyLoss()
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    if args.dataset == "cifar10":
        test_set = CIFAR10(
            "../data", train=False, transform=test_transform, download=True
        )
        test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    elif args.dataset == "cifar100":
        test_set = CIFAR100(
            "../data", train=False, transform=test_transform, download=True
        )
        test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    elif args.dataset == "restricted_imagenet":
        test_loader=  loader_
  
    correct = 0
    total_loss = 0
    total = 0  # number of samples
    num_batch = len(test_loader)
    use_cuda = True
    net.cuda()
    net.eval()
    with torch.no_grad():

        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                # print(inputs.size(0))
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()
    print("Correct %")
    print(100 * correct / total)
    misclassified = total - correct
    print("Total Loss")
    print(total_loss * 100 / total)
    print(f"misclassified samples from {total}")
    print(misclassified)

    return misclassified

def setup_model(args):

    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )

    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )

    elif args.dataset == "restricted_imagenet":
        classes =14
        
    if args.imagenet_arch:
        if args.dataset=="restricted_imagenet":
            classes = 14
        model = model_dict[args.arch](num_classes=classes, imagenet=True)

    else:
        model = model_dict[args.arch](num_classes=classes)
 
    if args.dataset!="restricted_imagenet":
        model.normalize = normalization

    return model

