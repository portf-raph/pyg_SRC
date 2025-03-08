import torch
from torch import Tensor

import numpy as np
from math import sqrt

from torch_geometric.utils import scatter
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj


def get_eigs(adj: Tensor):
    if adj.device == 'cuda':  # TODO: return type hint
        eigs, V = cp.linalg.eigh(cp.asarray(adj.to('cuda')))
        eigs, V = cp.squeeze(eigs), cp.squeeze(V)
        eigs = torch.Tensor(eigs)
        V = torch.Tensor(V)
    else:
        eigs, V = np.linalg.eigh(adj.detach().cpu().numpy())
        eigs, V = np.squeeze(eigs), np.squeeze(V)
        eigs = torch.from_numpy(eigs)
        V = torch.from_numpy(V)
    return eigs, V


def atol_eigs(eigs, edge_index) -> Tensor:
    row = edge_index[0]
    device = edge_index.device
    edge_weight = torch.ones(edge_index.shape[1], device=device)

    num_nodes = edge_index.max().item() + 1
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    return torch.ones(eigs.shape[0], device=device) - deg_inv_sqrt * eigs * deg_inv_sqrt


def norm_eigs(eigs) -> Tensor:
    return eigs / sqrt(eigs.shape[0])


def pyg_batch(data_dicts: list[dict]) -> Batch:
    batch = [Data(
        x=data_dict['x'],
        edge_index=data_dict['edge_index'],
    ) for data_dict in data_dicts]

    return Batch.from_data_list(batch, follow_batch=['x'])


def serial_routine(
                data: Data,
                upper: float,
                lower: float,
                count: int,
                logger
                ):
    data_dict = {}
    data_dict['edge_index'] = data.edge_index
    data_dict['edge_attr'] = data.edge_attr
    data_dict['x'] = data.x
    data_dict['y'] = data.y
    dense_adj = to_dense_adj(edge_index=data.edge_index,
                             edge_attr=data.edge_attr)
    eigs, V = get_eigs(dense_adj)

    data_dict['eigs'] = eigs
    if eigs is None:
        logger.info('eigs is None @ count {}'.format(count))
    eigs = norm_eigs(eigs)
    eigs_max = torch.max(eigs).item()
    eigs_min = torch.min(eigs).item()
    if eigs_max > upper:
        upper = eigs_max
    if eigs_min < lower:
        lower = eigs_min

    data_dict['V'] = V
    if V is None:
        logger.info('V is None @ count {}'.format(count))

    return data_dict, upper, lower
