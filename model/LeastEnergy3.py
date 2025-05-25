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
        # self.Q = torch.eye(num_atoms, device=device)
        
    def forward(self,
                _r_batch: Tensor,
                _D_batch: Tensor,
                _f_batch: Tensor):
        
        B = _f_batch.shape[0]
        N = _f_batch.shape[1]
        in_channels = _f_batch.shape[2]
        M = _D_batch.shape[2]  # self.num_atoms
        _f_batch = _f_batch.unsqueeze(0).expand(self.num_classes, -1, -1,-1).reshape(self.num_classes * B, N, in_channels)

        fid_batch = torch.sum(
            torch.square(_f_batch - torch.bmm(_D_batch, _r_batch).squeeze()), dim=1
        )
        fid_batch = torch.stack(torch.split(fid_batch.squeeze(), B)).T

        fid_batch_A = torch.sum(
            torch.square(_f_batch.detach() - torch.bmm(_D_batch, _r_batch).squeeze()), dim=1
        )
        fid_batch_A = torch.stack(torch.split(fid_batch_A.squeeze(), B)).T
        
        return fid_batch, fid_batch_A
