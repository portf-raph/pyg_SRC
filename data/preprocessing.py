import os
import math
import logging
import argparse
import numpy as np

import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

from utils.data_helper import get_eigs, serial_routine


def dump_eigs_data(
              save_dir: str,
              dataset_name: str,
              dataset: Dataset,
              seed: int,
              ):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for split_tag in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(save_dir, split_tag)):
            os.mkdir(os.path.join(save_dir, split_tag))

    log_file = os.path.join(save_dir, 'log_{}.txt'.format(dataset_name))
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger.info('Dumping eigs data from {}'.format(dataset_name))

    count = 0
    upper = 0
    lower = 0

    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.9* len(dataset))
    train_set = dataset[indices[:train_size]]
    test_set = dataset[indices[train_size:]]

    indices = torch.randperm(len(train_set)).tolist()
    sub_train_size = int(0.9*len(train_set))
    train_subset = train_set[indices[:sub_train_size]]
    val_set = train_set[indices[sub_train_size:]]

    for split in [train_subset, val_set, test_set]:
        split_tag = 'train' if split == train_subset else 'val' if split == val_set else 'test'
        for data in split:
            data_dict, upper, lower = serial_routine(
                data=data,
                upper=upper,
                lower=lower,
                count=count,
                logger=logger
            )
            torch.save(
                data_dict,
                open(
                    os.path.join(save_dir, split_tag, '{}_{}_{:05d}.pth'.format(dataset_name, split_tag, count)),
                    'wb'))
            if split_tag == 'train' and count == 0:
                logger.info('STORAGE: {}'.format(data.edge_index.device))
            count += 1
        logger.info('Done {} set'.format(split_tag))

    extreme_eigs = {
        'upper': upper,
        'lower': lower,
    }
    print(extreme_eigs)
    torch.save(
        extreme_eigs,
        open(os.path.join(save_dir, '{}_extreme_eigs.pth'.format(dataset_name)),
            'wb')
        )


def main():
    # 1.  Parser
    parser = argparse.ArgumentParser(
        description="Preprocessing data, getting pth file"
    )
    parser.add_argument('--save_root', type=str, default='./data',
                        help='Script config file path')
    parser.add_argument('--dataset_name', type=str, default='PROTEINS',
                        help='Name of TUDataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for random split')
    args = parser.parse_args()

    # 2. Pickel dump
    save_dir = os.path.join(args.save_root, args.dataset_name, 'pth')
    dataset = TUDataset(root=args.save_root, name=args.dataset_name)
    dump_eigs_data(
        save_dir=save_dir,
        dataset_name = args.dataset_name,
        dataset = dataset,
        seed = args.seed
        )

if __name__ == '__main__':
    main()
