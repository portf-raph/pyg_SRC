import math

import torch
from torch import Tensor

class LeastEnergy(torch.nn.Module):
    def __init__(self,
                 num_atoms: int,
                 num_classes: int,
                 partition: list[int],
                 device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.partition = partition
        assert self.partition[0] == 0
        assert self.partition[-1] == num_atoms
        assert self.num_classes + 1 == len(self.partition)
        self.Q = torch.eye(num_atoms, device=device)
        
    def forward(self,
                _r_batch: Tensor,
                _D_batch: Tensor,
                _f_batch: Tensor,):
        
        B = _f_batch.shape[0]
        N = _f_batch.shape[1]
        M = _D_batch.shape[2]  # self.num_atoms
        _f_batch = _f_batch.unsqueeze(0).expand(self.num_classes, -1, -1).reshape(self.num_classes * B, N)
        
        assert torch.allclose(_f_batch[0:B, :], _f_batch[B:2*B, :])
        assert _D_batch.shape[0] == _r_batch.shape[0]
        assert _D_batch.shape[1] == _f_batch.shape[1]
        assert _D_batch.shape[2] == _r_batch.shape[1]

        assert _f_batch.shape[0] == torch.bmm(_D_batch, _r_batch).squeeze().shape[0]
        assert _f_batch.shape[1] == torch.bmm(_D_batch, _r_batch).squeeze().shape[1]

        fid_batch = torch.sum(
            torch.square(_f_batch - torch.bmm(_D_batch, _r_batch).squeeze()), dim=1
        )
        fid_batch = torch.stack(torch.split(fid_batch.squeeze(), B)).T
        
        # === Position test ===
        _f_1 = _f_batch[0, :]
        _f_2 = _f_batch[0+B, :]
        assert torch.allclose(_f_1, _f_2)

        _D_1 = _D_batch[0, :, :].squeeze()
        _D_2 = _D_batch[0+B, :, :].squeeze()
        _r_1 = _r_batch[0, :, :].squeeze()
        _r_2 = _r_batch[0+B, :, :].squeeze()

        fid_1 = torch.sum(
                    torch.square(_f_1 - _D_1 @ _r_1)
                )
        fid_2 = torch.sum(
                    torch.square(_f_2 - _D_2 @ _r_2)
                )
        assert math.isclose(fid_batch[0, 0].item(), fid_1.item(), abs_tol=1e-4)
        assert math.isclose(fid_batch[0, 1].item(), fid_2.item(), abs_tol=1e-4)
        # === ///////////// ===
        
        return fid_batch
