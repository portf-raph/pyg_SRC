import time

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

from utils.data_helper import get_eigs, pyg_batch, atol_eigs

from data.pth_dataset import PthDataset
from model.GIN import GIN_Processor
from model.SparseCoder import SparseCoder


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(0)


def test_SparseCoder():
    dataset = PthDataset(load_dir='./data/PROTEINS/pth/train')
    data_dicts = []
    for i in range(8):
        data_dicts.append(dataset[i])

    data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                    else v for k, v in d.items()} for d in data_dicts]
    batch = pyg_batch(data_dicts)
    GIN = GIN_Processor(
        in_channels=3,
        num_layers=2,
        skip_first_features=True,
        MLP_cfg={'hid_dim': 12, 'num_hid': 2,
                 'dp_cfg': {'skip_first': True, 'dropout': 0.0},
                 'bn_cfg': {'use_batchnorm': True, 'batchnorm_affine': True},
                 'output_activation': 'relu',
                 },
        device=device
    )

    batch.x = GIN(batch.x, batch.edge_index)
    batch_vec = batch.x_batch
    for i in range(len(data_dicts)):
        assert data_dicts[i]['x'].shape[0] == torch.sum(batch_vec==i)
        data_dicts[i]['x'] = batch.x[batch_vec==i]

    GIN_out_channels = GIN.num_layers * 12
    SC = SparseCoder(
        K=5,
        in_channels=GIN_out_channels,
        num_atoms=45,
        num_classes=2,
        split_factor=2,
        _lambda=0.1,
        partition=[0, 15, 30, 45],
        _eta = 0.3,
        compute_loss=True,
        laplacian_eigs=True,
        pass_data=False,
        device=device
        )

    GIN_param = next(iter(GIN.parameters()))
    A_param = SC.A

    W = torch.randn(45, device=device)
    st = time.perf_counter()
    _r_batch, A_fidelity, A_incoherence, _D_batch, _f_batch = SC(data_dicts)
    loss = torch.sum(_r_batch.squeeze() @ W)
    loss.backward()
    et = time.perf_counter()

    print("GIN param grad: {}".format(GIN_param.grad))
    print("A_fidelity: {}, \n A_incoherence: {}".format(A_fidelity, A_incoherence))
    print('{}s elapsed'.format(et-st))

    return 0
