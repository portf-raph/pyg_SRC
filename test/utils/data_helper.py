import os
import time

import torch
import torch_geometric
from torch_geometric.utils import scatter, to_dense_adj
from torch_geometric.data import Dataset

from utils.random_graph import er_graph
from utils.data_helper import *


def test_get_eigs(edge_index):
    if edge_index is None:
        torch.manual_seed(1)
        edge_index = er_graph(num_nodes=20, num_edges=200)    # mod call
        edge_index = torch.Tensor(edge_index).type(torch.int64)
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.ones(edge_index.shape[1])
    adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)

    # 1. No GPU
    adj.to('cpu')
    st = time.time()
    eigs_cpu, V_cpu = get_eigs(adj)   # mod call
    et = time.time()
    print('No GPU: {}s elapsed'.format(et-st))

    # 2. GPU
    adj.to('cuda')
    torch.cuda.is_available = lambda : True
    st = time.time()
    eigs_gpu, V_gpu = get_eigs(adj)   # mod call
    et = time.time()
    print('GPU: {}s elapsed'.format(et-st))
    assert torch.allclose(eigs_cpu, eigs_gpu)
    assert torch.allclose(V_cpu, V_gpu)

    return 0


def test_atol_eigs(edge_index):
    if edge_index is None:
        torch.manual_seed(1)
        edge_index = er_graph(num_nodes=20, num_edges=200)    # mod call
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.ones(edge_index.shape[1]).to(edge_index.device)
    row = edge_index[0]

    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    eigs = torch.randn(deg.shape[0])

    result = atol_eigs(eigs, edge_index)    # mod call
    result_valid = torch.eye(eigs.shape[0]) - torch.diag(deg_inv_sqrt) @ torch.diag(eigs) @ torch.diag(deg_inv_sqrt)
    assert torch.equal(torch.diag(result), result_valid)

    return 0
