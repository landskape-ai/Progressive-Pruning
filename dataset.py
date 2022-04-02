"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision
__all__ = [
    "cifar10_dataloaders",
    "cifar100_dataloaders",
    "generate_anytime_cifar10_dataloader",
]






from robustness.datasets import RestrictedImageNetBalanced






def to_few_shot(dataset, n_shots=10):
    """
    Transforms torchvision dataset to a few-shot dataset.
    :param dataset: torchvision dataset
    :param n_shots: number of samples per class
    :return: few-shot torchvision dataset
    """
    try:
        targets = dataset.targets  # targets or labels depending on the dataset
        is_targets = True
    except:
        targets = dataset.labels
        is_targets = False

    assert min(targets) == 0, 'labels should start from 0, not from {}'.format(min(targets))

    # Find n_shots samples for each class
    labels_dict = {}
    imgs = dataset.imgs
    for i, lbl in enumerate(imgs):
        if lbl[1] not in labels_dict:
            labels_dict[lbl[1]] = []
        if len(labels_dict[lbl[1]]) < n_shots:
            labels_dict[lbl[1]].append(i)
            

    idx = sorted(torch.cat([torch.tensor(v) for k, v in labels_dict.items()]))  # sort according to the original order in the full dataset
    dataset.imgs = [dataset.imgs[i] for i in idx] if isinstance(dataset.imgs, list) else dataset.imgs[idx]
    targets = [imgs[i][1] for i in idx]
    if is_targets:
        dataset.targets = targets
    else:
        dataset.labels = targets

    return dataset



def Setup_RestrictedImageNet(args,path):
    ds = RestrictedImageNetBalanced(path)
  
    train_set, test_set = ds.make_loaders(batch_size=128, workers=8)

    if args.few_shot:
        print("Few Shot Regime Train Data Loading ")
        train_set =to_few_shot(train_set,n_shots= args.n_shots)
    
    return train_set,test_set


def generate_anytime_res_img_dataloader_few(args, whole_trainset,test_set,sample_len,state=1):
    
    meta_train_size = int(args.meta_batch_size * 0.9)  # 29839# 
    meta_val_size = args.meta_batch_size - meta_train_size  # 500

    if args.no_replay:
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(sample_len + (state - 1) * meta_val_size, sample_len + state * meta_val_size)
        )
    elif args.one_replay:
        if state ==1:
            train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
            val_list = list(
                range(sample_len + (state - 1) * meta_val_size, sample_len + state * meta_val_size)
            )
        else: 
            train_list = list(range((state - 2) * meta_train_size, state * meta_train_size))
            val_list = list(
                range(sample_len + (state - 2) * meta_val_size, sample_len + state * meta_val_size)
            )
    elif args.buffer_replay:
        k = args.buffer_size_train
        l = args.buffer_size_val
        
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(sample_len + (state - 1) * meta_val_size, sample_len + state * meta_val_size)
        )

        train_list.extend(buffer_train_set)
        val_list.extend(buffer_val_set)
        
        # Populating Buffer         
        train_sampled_set = random.sample(train_list,k)
        valid_sampled_set = random.sample(val_list,l)
        
        buffer_train_set.extend(train_sampled_set)
        buffer_val_set.extend(valid_sampled_set)

    else:
        train_list = list(range(0, state * meta_train_size))  # 0 45000
        val_list = list(range(sample_len, sample_len + state * meta_val_size))  # 45000 500

    print(
        "Current: Trainset size = {}, Valset size = {}".format(
            len(train_list), len(val_list)
        )
    )

    train_set = Subset(whole_trainset, train_list)
    val_set = Subset(whole_trainset, val_list)


    if args.snip_no_replay:
        train_list_norep = list(range((state - 1) * meta_train_size, state * meta_train_size))
        train_set_norep = Subset(whole_trainset, train_list_norep)
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set_norep, list(range(snip_set)))
    else:
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set, list(range(snip_set)))
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_snip_set



def generate_anytime_res_img_dataloader(args, whole_trainset,test_set, sample_len,state=1):

    meta_train_size = int(args.meta_batch_size * 0.9)  # 29839#
    meta_val_size = args.meta_batch_size - meta_train_size  # 500

    if args.no_replay:
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(sample_len + (state - 1) * meta_val_size, sample_len + state * meta_val_size)
        )
    elif args.one_replay:
        if state ==1:
            train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
            val_list = list(
                range(sample_len + (state - 1) * meta_val_size, sample_len + state * meta_val_size)
            )
        else: # 0-1, 1-2,2-3,3-4,4-5 
            train_list = list(range((state - 2) * meta_train_size, state * meta_train_size))
            val_list = list(
                range(sample_len + (state - 2) * meta_val_size, sample_len + state * meta_val_size)
            )
    elif args.buffer_replay:
        k = args.buffer_size_train
        l = args.buffer_size_val
        
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(sample_len + (state - 1) * meta_val_size, sample_len + state * meta_val_size)
        )

        train_list.extend(buffer_train_set)
        val_list.extend(buffer_val_set)
        
        # Populating Buffer         
        train_sampled_set = random.sample(train_list,k)
        valid_sampled_set = random.sample(val_list,l)
        
        buffer_train_set.extend(train_sampled_set)
        buffer_val_set.extend(valid_sampled_set)

    else:
        train_list = list(range(0, state * meta_train_size))  # 0 45000
        val_list = list(range(sample_len, sample_len + state * meta_val_size))  # 45000 500

    print(
        "Current: Trainset size = {}, Valset size = {}".format(
            len(train_list), len(val_list)
        )
    )

    train_set = Subset(whole_trainset, train_list)
    val_set = Subset(whole_trainset, val_list)


    if args.snip_no_replay:
        train_list_norep = list(range((state - 1) * meta_train_size, state * meta_train_size))
        train_set_norep = Subset(whole_trainset, train_list_norep)
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set_norep, list(range(snip_set)))
    else:
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set, list(range(snip_set)))
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_snip_set


def cifar10_dataloaders(batch_size=128, data_dir="datasets/cifar10", num_workers=2):

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = Subset(
        CIFAR10(data_dir, train=True, transform=train_transform, download=True),
        list(range(45000)),
    )
    val_set = Subset(
        CIFAR10(data_dir, train=True, transform=test_transform, download=True),
        list(range(45000, 50000)),
    )
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(batch_size=128, data_dir="datasets/cifar100", num_workers=2):

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = Subset(
        CIFAR100(data_dir, train=True, transform=train_transform, download=True),
        list(range(45000)),
    )
    val_set = Subset(
        CIFAR100(data_dir, train=True, transform=test_transform, download=True),
        list(range(45000, 50000)),
    )
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def setup__cifar10_dataset(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


    whole_trainset = CIFAR10(
        args.data, train=True, transform=train_transform, download=True
    )
    return whole_trainset

def setup__cifar10_dataset_end(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    whole_trainset = CIFAR10(
        args.data, train=True, transform=train_transform, download=True
    )
    #50,000 -200 = 49800 
    end_list = list(range(49800, 50000))
    sub_whole_trainset = Subset(whole_trainset, list(range(49800)))
    end_trainset = Subset(whole_trainset, end_list)

    return sub_whole_trainset, end_trainset


def generate_anytime_cifar10_dataloader_end(args, whole_trainset, state=1):

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    #45000-200 = 49800 , 49800-623  
    meta_train_size = int(args.meta_batch_size * 0.9)  #  #5602
    meta_val_size = args.meta_batch_size - meta_train_size  # 623

    if args.no_replay:
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(44816 + (state - 1) * meta_val_size, 44816 + state * meta_val_size)
        )
    else:
        train_list = list(range(0, state * meta_train_size))  # 0 44816
        val_list = list(range(44816, 44816 + state * meta_val_size))  # 45000 500

    print(
        "Current: Trainset size = {}, Valset size = {}".format(
            len(train_list), len(val_list)
        )
    )

    train_set = Subset(whole_trainset, train_list)
    val_set = Subset(whole_trainset, val_list)
    test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)

    if args.snip_no_replay:
        train_list_norep = list(range((state - 1) * meta_train_size, state * meta_train_size))
        train_set_norep = Subset(whole_trainset, train_list_norep)
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set_norep, list(range(snip_set)))
    else:
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set, list(range(snip_set)))
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_snip_set

buffer_train_set = []
buffer_val_set = []

def generate_anytime_cifar10_dataloader(args, whole_trainset, state=1):

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    meta_train_size = int(args.meta_batch_size * 0.9)  # 4500 #
    meta_val_size = args.meta_batch_size - meta_train_size  # 500

    if args.no_replay:
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(45000 + (state - 1) * meta_val_size, 45000 + state * meta_val_size)
        )
    elif args.one_replay:
        if state ==1:
            train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
            val_list = list(
                range(45000 + (state - 1) * meta_val_size, 45000 + state * meta_val_size)
            )
        else: # 0-1, 1-2,2-3,3-4,4-5 
            train_list = list(range((state - 2) * meta_train_size, state * meta_train_size))
            val_list = list(
                range(45000 + (state - 2) * meta_val_size, 45000 + state * meta_val_size)
            )
    elif args.buffer_replay:
        k = args.buffer_size_train
        l = args.buffer_size_val
        
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(45000 + (state - 1) * meta_val_size, 45000 + state * meta_val_size)
        )

        train_list.extend(buffer_train_set)
        val_list.extend(buffer_val_set)
        
        # Populating Buffer         
        train_sampled_set = random.sample(train_list,k)
        valid_sampled_set = random.sample(val_list,l)
        
        buffer_train_set.extend(train_sampled_set)
        buffer_val_set.extend(valid_sampled_set)

    else:
        train_list = list(range(0, state * meta_train_size))  # 0 45000
        val_list = list(range(45000, 45000 + state * meta_val_size))  # 45000 500

    print(
        "Current: Trainset size = {}, Valset size = {}".format(
            len(train_list), len(val_list)
        )
    )

    train_set = Subset(whole_trainset, train_list)
    val_set = Subset(whole_trainset, val_list)
    test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)

    if args.snip_no_replay:
        train_list_norep = list(range((state - 1) * meta_train_size, state * meta_train_size))
        train_set_norep = Subset(whole_trainset, train_list_norep)
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set_norep, list(range(snip_set)))
    else:
        snip_set = int(args.meta_batch_size * args.snip_size)
        train_snip_set = Subset(train_set, list(range(snip_set)))
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_snip_set


def setup__cifar100_dataset(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    whole_trainset = CIFAR100(
        args.data, train=True, transform=train_transform, download=True
    )

    return whole_trainset


def generate_anytime_cifar100_dataloader(args, whole_trainset, state=1):

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    meta_train_size = int(args.meta_batch_size * 0.9)  # 4500
    meta_val_size = args.meta_batch_size - meta_train_size  # 500

    if args.no_replay:
        train_list = list(range((state - 1) * meta_train_size, state * meta_train_size))
        val_list = list(
            range(45000 + (state - 1) * meta_val_size, 45000 + state * meta_val_size)
        )
    else:
        train_list = list(range(0, state * meta_train_size))  # 0 45000
        val_list = list(range(45000, 45000 + state * meta_val_size))  # 45000 500

    print(
        "Current: Trainset size = {}, Valset size = {}".format(
            len(train_list), len(val_list)
        )
    )

    train_set = Subset(whole_trainset, train_list)
    val_set = Subset(whole_trainset, val_list)

    test_set = CIFAR100(args.data, train=False, transform=test_transform, download=True)

    snip_set = int(args.meta_batch_size * args.snip_size)
    train_snip_set = Subset(train_set, list(range(snip_set)))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_snip_set
