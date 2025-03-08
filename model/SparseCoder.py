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
                 laplacian_eigs: bool=True,
                 compute_loss: bool=True,
                 pass_data: bool=False,
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
        self.compute_loss = compute_loss
        self.pass_data = pass_data
        self.device = device

        self.split_factor = split_factor
        self.padding = self.num_atoms - 2*partition[1]    # TEMP
        self._f_split = build_f_split(split_factor=split_factor)
        self._f_mend = build_f_mend(in_channels=in_channels,
                                    split_factor=split_factor,
                                    padding=self.padding)
        self._f_mend_0 = build_f_mend(in_channels=self.in_channels,
                                      split_factor=self.split_factor,
                                      padding=0)
        self._f_mend_alt = build_f_mend_alt(in_channels=in_channels)

        self.A = Parameter(torch.randn(self.K, self.in_channels*self.num_atoms, device=device))  # TODO: A init

        # Selection operators
        assert self.partition[0] == 0
        assert self.partition[-1] == num_atoms
        assert self.num_classes + 2 == len(self.partition)
        self.Q = torch.eye(self.num_atoms, device=device)

    def forward(self,
                data_dicts: list[dict]):
        _D_batch = [get_dict(A=self.A,
                             K=self.K,
                             in_channels=self.in_channels,
                             V=data_dict['V'],
                             eigs=atol_eigs(data_dict['eigs'], data_dict['edge_index']) if self.laplacian_eigs else norm_eigs(data_dict['eigs']))
                             for data_dict in data_dicts]
        _D_batch = torch.stack(
                pad_columns(_D_batch)
              )
        y_batch = [data_dict['y'].item() for data_dict in data_dicts]
        start_batch = [self.partition[y] for y in y_batch]
        end_batch = [self.partition[y+1] for y in y_batch]
        
        _f_stack = [data_dict['x'] for data_dict in data_dicts]
        _f_stack = self._f_split(_f_stack)    # SPLIT
        
        if self.training:
            A_fidelity = 0
            A_incoherence = 0

            if self.compute_loss:
                _D_norm = F.normalize(_D_batch, p=2, dim=1, eps=1e-12)
                _D_norm = torch.unbind(_D_norm)
                _D_label_batch = torch.stack(
                    [_D_norm[i][:, start:end] for i, (start, end) in enumerate(zip(start_batch, end_batch))]
                )
                _D_rest_batch = torch.stack(
                    [torch.cat((_D_norm[i][:, 0:start],
                                _D_norm[i][:, end:self.partition[-1]]), dim=1) for i, (start, end) in enumerate(zip(start_batch, end_batch))]
                ) # TEMP
                A_incoherence = (1/len(data_dicts)) * torch.sum(torch.square(torch.bmm(_D_label_batch.transpose(-1,-2), _D_rest_batch)))

            sub_Q_batch = torch.stack(
                [torch.cat((self.Q[:, start:end],
                            self.Q[:, self.partition[-2]:self.partition[-1]]), dim=1) for start, end in zip(start_batch, end_batch)]
            )
            sub_D_batch = torch.bmm(_D_batch,
                                    torch.bmm(sub_Q_batch, sub_Q_batch.transpose(-1,-2))
                                    ) 
            Q_ex_T_batch = torch.stack(
                [torch.cat((self.Q[:, 0:start],
                            self.Q[:, end:self.partition[-2]]), dim=1).T for start, end in zip(start_batch, end_batch)]
            )
            _D_stack = torch.cat([_D_batch, sub_D_batch, Q_ex_T_batch], dim=1).contiguous()

            _r_batch = FISTA_bmm(None,
                                _f_stack=_f_stack,
                                _D_stack=_D_stack,
                                _lambda=self._lambda,
                                _f_mend=self._f_mend,
                                )
            _r_batch = torch.stack(_r_batch)
            _f_batch = self._f_mend_alt(_f_stack, 
                                        _D_stack.shape[0]).detach()
            
            if self.compute_loss:
                _D_fid = _D_stack[:, 0:-self.padding, :]
                _f_fid =  torch.cat([_f_batch, _f_batch], dim=1)
                A_fidelity = (1/len(data_dicts)) * (1/2) * torch.sum(torch.square(_f_fid - torch.bmm(_D_fid, _r_batch.detach()).squeeze()))

            return _r_batch, A_fidelity, A_incoherence, _D_batch.detach(), _f_batch

        else:
            _r_batch = FISTA_bmm(None,
                                _f_stack=_f_stack,
                                _D_stack=_D_batch,
                                _lambda=self._lambda,
                                _f_mend=self._f_mend_alt,
                                )
            _r_batch = torch.stack(_r_batch)
            _f_batch = self._f_mend_alt(_f_stack, 
                                        _D_batch.shape[0]).detach()
            return _r_batch, None, None, _D_batch.detach(), _f_batch
