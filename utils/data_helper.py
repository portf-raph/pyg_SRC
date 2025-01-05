from torch import Tensor
import cupy as cp
import numpy as np

from torch_geometric.utils import scatter
from torch_geometric.data import Batch, Data


def get_eigs(adj: Tensor):
    if adj.device == 'cuda':
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
    edge_weight = torch.ones(edge_index.shape[1]).to(edge_index.device)

    num_nodes = edge_index.max().item() + 1
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    return torch.ones(eigs.shape[0]) - deg_inv_sqrt * eigs * deg_inv_sqrt


def pyg_batch(data_dicts: list[dict]) -> Batch:
    batch = []
    for data_dict in data_dicts:
        graph = Data(
            x=data_dict['x'],
            edge_index=data_dict['edge_index'],
          )
        batch.append(graph)
    return Batch.from_data_list(batch, follow_batch=['x'])
