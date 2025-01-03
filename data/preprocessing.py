import os
import pickle
import argparse
import logging

from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

from utils.data_helper import get_eigs


def dump_eigs_data(
              save_dir: str,
              ds_name: str,
              tag: str,
              dataset: Dataset,
              ):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(save_dir + '/pkl'):
        os.mkdir(save_dir + '/pkl')

    log_file = os.path.join(save_dir, 'log_{}.txt'.format(ds_name))
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger.info('Dumping eigs data from {}'.format(tag))

    count = 0
    for data in dataset:
        data_dict = {}
        data_dict['edge_index'] = data.edge_index
        data_dict['edge_attr'] = data.edge_attr
        data_dict['x'] = data.x
        data_dict['y'] = data.y

        dense_adj = to_dense_adj(edge_index=data.edge_index,
                                 edge_attr=data.edge_attr)
        eigs, V = get_eigs(dense_adj)    # mod call
        if count == 0:
            logger.info('STORAGE: {}'.format(data.edge_index.device))
        data_dict['eigs'] = eigs
        if eigs is None:
            logger.info('eigs is None @ count {}'.format(count))
        data_dict['V'] = V
        if V is None:
            logger.info('V is None @ count {}'.format(count))

        pickle.dump(
        data_dict,
        open(
            os.path.join(save_dir, 'pkl/{}_{}_{:07d}.p'.format(ds_name, tag, count)),
            'wb'))

        count += 1

    logger.info('Done')


def main():
    # 1.  Parser
    parser = argparse.ArgumentParser(
        description="Preprocessing data, getting pkl file"
    )
    parser.add_argument('--save_dir', type=str, default='../data',
                        required=True, help='Script config file path')
    parser.add_argument('--ds_name', type=str, default='PROTEINS',
                        required=True, help='Name of TUDataset')
    parser.add_argument('--tag', type=str, default='train',
                        required=True, help='Additional dataset tag')

    args = parser.parse_args()

    # 2. Pickel dump
    dataset = TUDataset(root='../data', name=args.ds_name)
    dump_eigs_data(
        save_dir=args.save_dir,
        ds_name = args.ds_name,
        tag = args.tag,
        dataset=dataset)

if __name__ == '__main__':
    main()
