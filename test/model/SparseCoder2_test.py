import torch
import itertools
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils.dict_learning import *
from utils.data_helper import atol_eigs, norm_eigs
from utils.FISTA import FISTA_bmm

class SparseCoder(torch.nn.Module):
    def __init__(self,
                 K: int,
                 in_channels: int,  # C
                 num_atoms: int,  # M
                 num_classes: int,
                 split_factor: int,
                 _lambda: float,
                 _eta: float,
                 partition: list[int],
                 laplacian_eigs: bool=False,
                 device='cpu'
                 ):

        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.num_atoms = num_atoms
        self.num_classes = num_classes
        self.partition = partition
        self._lambda = _lambda
        self._eta = _eta
        self.laplacian_eigs = laplacian_eigs
        self.device = device

        self.split_factor = split_factor
        self._f_split = build_f_split(split_factor=split_factor)
        self._f_mend_repeat = build_f_mend_repeat(
                num_classes=num_classes,
                in_channels=in_channels)  # d
        self._f_mend_alt = build_f_mend_alt(in_channels=in_channels)

        self.A = Parameter(torch.randn(self.K, self.in_channels*self.num_atoms, device=device))  # TODO: A init

        # Selection operators
        assert self.partition[0] == 0
        assert self.partition[-1] == num_atoms
        assert self.num_classes + 1 == len(self.partition)  # d
        self.Q = torch.eye(self.num_atoms, device=device)

    def forward(self,
                data_dicts: list[dict]):
        A_fidelity = 0
        A_incoherence = 0

        _D_batch = [get_dict(A=self.A,
                             K=self.K,
                             in_channels=self.in_channels,
                             V=data_dict['V'],
                             eigs=atol_eigs(data_dict['eigs'], data_dict['edge_index']) if self.laplacian_eigs else data_dict['eigs'])
                             for data_dict in data_dicts]
        _D_batch = torch.stack(
                pad_columns(_D_batch)
                )
        _D_batch_0 = torch.cat(
                    torch.split(_D_batch, self.partition[1], dim=-1),dim=0
                )   # d
        y_batch = [data_dict['y'].item() for data_dict in data_dicts]
        start_batch = [self.partition[y] for y in y_batch]
        end_batch = [self.partition[y+1] for y in y_batch]
        batch_size = len(y_batch)
        
        _f_stack = [data_dict['x'] for data_dict in data_dicts]
        _f_stack = self._f_split(_f_stack)    # SPLIT
        _r_batch_0 = FISTA_bmm(None,
                             _f_stack=_f_stack,
                             _D_stack=_D_batch_0,
                             _lambda=self._lambda,
                             _f_mend=self._f_mend_repeat,
                             )
        _r_batch_0 = torch.stack(_r_batch_0)
        _f_batch = self._f_mend_alt(_f_stack,
                                    batch_size)

        # === Split FISTA validation ===
        _D_batch_1 = _D_batch[:, :, 0:self.partition[1]].detach()
        _D_batch_2 = _D_batch[:, :, self.partition[1]:self.partition[2]].detach()
        _r_batch_1 = FISTA_bmm(None,
                               _f_stack=_f_stack,
                               _D_stack=_D_batch_1,
                               _lambda=self._lambda,
                               _f_mend=self._f_mend_alt
                               )
        _r_batch_1 = torch.stack(_r_batch_1)
        _r_batch_2 = FISTA_bmm(None,
                               _f_stack=_f_stack,
                               _D_stack=_D_batch_2,
                               _lambda=self._lambda,
                               _f_mend=self._f_mend_alt
                               )
        _r_batch_2 = torch.stack(_r_batch_2)
        assert torch.allclose(_r_batch_1, _r_batch_0[0:batch_size], atol=1e-6)
        assert torch.allclose(_r_batch_2, _r_batch_0[batch_size:2*batch_size], atol=1e-6)
        # === /////////////////// ===
            
        if self.training:
            _r_batch = torch.cat(
                            torch.split(_r_batch_0, batch_size, dim=0), dim=1
                         ).detach() # (n_c x B x M/n_c x 1) -> (B x M x 1)

            # === Fidelity ===
            sub_Q_batch = torch.stack(
                [self.Q[:, start:end] for start, end in zip(start_batch, end_batch)]
            )
            sub_D_batch = torch.bmm(_D_batch,
                                    torch.bmm(sub_Q_batch, sub_Q_batch.transpose(-1,-2))
                                    )
            _D_fid = sub_D_batch     # d
            _f_fid = _f_batch.detach()   # d
            A_fidelity = (1/len(data_dicts)) * (1/2) * torch.sum(torch.square(_f_fid - torch.bmm(_D_fid, _r_batch).squeeze()))
            # === //////// ===

            # === Incoherence ===
            _D_norm = F.normalize(_D_batch.detach(), p=2, dim=1, eps=1e-12)
            _D_norm = torch.unbind(_D_norm)
            _D_label_batch = torch.stack(
                [_D_norm[i][:, start:end] for i, (start, end) in enumerate(zip(start_batch, end_batch))]
            )
            _D_rest_batch = torch.stack(
                [torch.cat((_D_norm[i][:, 0:start],
                            _D_norm[i][:, end:self.partition[-1]]), dim=1) for i, (start, end) in enumerate(zip(start_batch, end_batch))]
                ) # TEMP
            A_incoherence = (1/len(data_dicts)) * torch.sum(torch.square(torch.bmm(_D_label_batch.transpose(-1,-2), _D_rest_batch)))
            # === ////////// ===

        return _r_batch_0.detach(), A_fidelity, A_incoherence, _D_batch_0, _f_batch
