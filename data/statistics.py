# VAL
import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Dataset

from .pth_dataset import PthDataset


def run_stats(
    save_dir: str,
    dataset_name: str,
    train_set: Dataset,
    extreme_eigs: dict,
    num_classes: int,
    num_bounds: int=100,
    ):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_file = os.path.join(save_dir, 'log_{}.txt'.format(dataset_name))
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running statistics on training set')

    class_eigs = []
    num_eigs = [0] * num_classes

    upper = extreme_eigs['upper']
    lower = extreme_eigs['lower']
    bounds = np.linspace(upper, lower, num_bounds)
    bin_length = bounds[1] - bounds[0]
    bin_centers = bounds + bin_length/2   # val
    torch.save(
        bin_centers,
        open(os.path.join(save_dir, '{}_bin_centers.pth'.format(dataset_name)),
            'wb')
    )

    class_bins = [[0] * (num_bounds) for i in range(num_classes)]
    for data_dict in train_set:
        eigs = norm_eigs(data_dict["eigs"]).numpy()
        label = data_dict["y"]
        num_eigs[label] += eigs.shape[0]  # val

        unique_bins, bin_counts = np.unique(np.digitize(eigs, bounds), return_counts=True)
        for bin_idx, bin_count in zip(unique_bins, bin_counts):
            class_bins[label][bin_idx-1] += bin_count
        if label == 0:
            class_eigs += eigs.tolist()

    # Plot eigs of class 0
    plt.figure(figsize=(10,1))
    plt.scatter(class_eigs, np.zeros(len(class_eigs)), marker='x', alpha=0.7, linewidths=0.01)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.yticks([])
    plt.xlabel('Spectral interval')
    plt.title('Spectral density of class 0')
    plt.show()

    for i in range(num_classes):
        class_bins[i] = np.array(class_bins[i]) / num_eigs[i]
        class_bins[i][class_bins[i]==0] = 1e-5  # temporary fix
        plt.bar(bin_centers, class_bins[i], label='Class {} spectral density'.format(i), width=bin_length)
        plt.show()
    class_weights = [0] * num_classes
    print('=================================== Spectral densities ===================================')

    for i in range(num_classes):
        complementary = np.stack([class_bin for j,class_bin in enumerate(class_bins) if j != i])
        class_weights[i] = np.divide(
                np.mean(complementary, axis=0), class_bins[i]
            )
        plt.bar(bin_centers, class_weights[i], label='Class {} weights'.format(i), width=bin_length)
        plt.show()
        class_weights[i] = torch.from_numpy(class_weights[i])

    print('=================================== Weight factors ===================================')

    torch.save(
        torch.stack(class_weights),
        open(os.path.join(save_dir, '{}_class_weights.pth'.format(dataset_name)),
             'wb')
    )
    logger.info('Done running statistics')


def main():
    # 1.  Parser
    parser = argparse.ArgumentParser(
        description="Preprocessing data, getting pth file"
    )
    parser.add_argument('--save_root', type=str, default='./data',
                        required=True, help='Script config file path')
    parser.add_argument('--dataset_name', type=str, default='PROTEINS',
                        required=True, help='Name of TUDataset')
    parser.add_argument('--num_bounds', type=int, default=100,
                        required=True, help='Number of bins, plus 1')
    args = parser.parse_args()

    # 2. Pickel dump
    save_dir = os.path.join(args.save_root, args.dataset_name, 'pth')
    dataset = TUDataset(root=args.save_root, name=args.dataset_name)
    num_classes = dataset.num_classes
    train_set = PthDataset(load_dir=os.path.join(save_dir, 'train'))
    extreme_eigs = torch.load(
        open(os.path.join(save_dir, '{}_extreme_eigs.pth'.format(args.dataset_name)),
             'rb')
    )
    run_stats(
        save_dir=save_dir,
        dataset_name=args.dataset_name,
        train_set=train_set,
        extreme_eigs=extreme_eigs,
        num_bounds=args.num_bounds,
        num_classes=num_classes
        )


if __name__ == '__main__':
    main()
