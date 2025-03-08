import itertools
from collections.abc import Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torchopt.typing import TupleOfTensors, TensorOrTensors


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dict(A: Tensor,   # K x CM
             V: TupleOfTensors,
             eigs: TupleOfTensors,
             K: int,
             in_channels: int,
             ) -> TupleOfTensors:

    assert (A.shape[1] % in_channels == 0)

    G_k = torch.stack([torch.pow(eigs, k) for k in range(K)]).T
    D = G_k @ A
    D = V @ D

    num_nodes = V.shape[0]
    D = D.t().flatten()
    D = D.split(num_nodes * in_channels, dim=0)
    D = torch.stack(D, dim=0)
    D = D.t()

    return D    # TODO: bottleneck, but parallelizing is too much of a hassle


def scode_obj_bmm(params: TupleOfTensors,
                  _f_stack: TupleOfTensors,
                  _D_stack: Tensor,
                  _lambda: float,
                  _f_mend: Callable
                  ):

    _f_stack = _f_mend(_f_stack=_f_stack,
                       batch_size=_D_stack.shape[0]).unsqueeze(-1)

    params = torch.stack(params)
    regress = (1/2) * torch.sum(torch.square(_f_stack - torch.bmm(_D_stack, params)), dim=1)
    penalty = _lambda * torch.linalg.norm(params.squeeze(), ord=1, dim=1)
    return torch.sum(regress + penalty) / params.shape[0]


def get_R(
    V: Tensor,
    eigs: Tensor,
    _r: Tensor,
    K: int,
    ):
    M = _r.shape[0]
    G_k = torch.stack([torch.pow(eigs, k) for k in range(K)]).T
    VG_k = V @ G_k

    VG_k_expanded = VG_k.unsqueeze(0).expand(M, -1, -1)
    R_r = torch.einsum('ijk,i->ijk', VG_k_expanded, _r).transpose(0,1).reshape(V.shape[0], -1)

    return R_r


def get_OmegaP(
    class_weights: Tensor,
    bin_centers: Tensor,
    partition: list,
    K: int,
    ):

    partition.pop()
    P = torch.stack([torch.pow(bin_centers,k) for k in range(K)]).T   # H x K
    OmegaP = torch.cat(
        [(torch.diag(class_weights[i, :]) @ P).repeat(1, partition[i+1]-partition[i])
        for i in range(len(partition)-1)],
                       dim=1)
    return OmegaP


def build_f_split(split_factor: int):

    def _f_split(_f_stack: list[Tensor]):
        """
        _f_stack = [torch.tensor_split(_f, split_factor, dim=0) for _f in _f_stack]  # [B] -> [B, [sp]]
        _f_stack = list(itertools.chain.from_iterable(_f_stack))    # [B*sp]
        _f_stack = [torch.unbind(_f, dim=1) for _f in _f_stack]     # [B*sp, (C)]
        _f_stack = tuple(itertools.chain.from_iterable(_f_stack))   # (B*sp*C)
        """
        _f_stack = [torch.unbind(_f, dim=1) for _f in _f_stack]     
        _f_stack = tuple(itertools.chain.from_iterable(_f_stack))   
        return _f_stack

    return _f_split


def build_f_mend(in_channels: int,
                 split_factor: int,
                 padding: int):

    def _f_mend(_f_stack: TupleOfTensors,
                batch_size: int):
        """
        _f_stack = [torch.stack(
            _f_stack[i*in_channels:(i+1)*in_channels], dim=1
            ) for i in range(batch_size * split_factor)
          ]
        # (B*sp*C) -> [B*sp] (Nv/sp x C)
        _f_stack = [torch.cat(
            _f_stack[i*split_factor:(i+1)*split_factor], dim=0
            ) for i in range(batch_size)
        ]
        # [B] (Nv x C)
        _f_stack = [torch.unbind(_f, dim=1) for _f in _f_stack]
        _f_stack = tuple(itertools.chain.from_iterable(_f_stack))
        """

        _f_stack = [torch.cat(
            _f_stack[i*in_channels:(i+1)*in_channels]
            ).unsqueeze(-1) for i in range(batch_size)
          ]

        _f_stack = torch.stack(
            pad_columns(_f_stack)
            ).squeeze()
        
        _f_stack = F.pad(
            torch.cat([_f_stack, _f_stack], dim=1), (0, padding)
        )

        return _f_stack

    return _f_mend


def build_f_mend_alt(in_channels: int):

    def _f_mend_alt(_f_stack: TupleOfTensors,
                    batch_size):

        _f_stack = [torch.cat(
                _f_stack[i*in_channels:(i+1)*in_channels]
            ).unsqueeze(-1) for i in range(batch_size)
          ]

        _f_stack = torch.stack(
                pad_columns(_f_stack)
            ).squeeze()

        return _f_stack

    return _f_mend_alt


def pad_columns(tensors):
    lengths = [tensor.shape[0] for tensor in tensors]
    max_length = max(lengths)
    padded_tensors = [F.pad(tensor, (0, 0, 0, max_length - lengths[i]))
                      for i, tensor in enumerate(tensors)]
    return padded_tensors
