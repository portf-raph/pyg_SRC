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
        assert self.num_classes + 2 == len(self.partition)
        self.Q = torch.eye(num_atoms, device=device)
        
    def forward(self,
                _r_batch: Tensor,
                _D_batch: Tensor,
                _f_batch: Tensor,):
        
        B = _D_batch.shape[0]
        N = _D_batch.shape[1]
        M = _D_batch.shape[2]  # self.num_atoms
        
        start_batch = [self.partition[i] for i in range(self.num_classes)]
        end_batch = [self.partition[i+1] for i in range(self.num_classes)]
        
        sub_Q_batch = [torch.cat((self.Q[:, start:end], 
                                  self.Q[:, self.partition[-2]:self.partition[-1]]), dim=1) for start, end in zip(start_batch, end_batch)]
        Q_prod_batch = [sub_Q @ sub_Q.T for sub_Q in sub_Q_batch]
        Q_prod_batch = torch.cat(
            [Q_prod.unsqueeze(0).expand(B, -1, -1) for Q_prod in Q_prod_batch], dim=0
        )

        _D_batch = _D_batch.unsqueeze(0).expand(self.num_classes, -1, -1, -1).reshape(self.num_classes * B, N, M)
        _f_batch = _f_batch.unsqueeze(0).expand(self.num_classes, -1, -1).reshape(self.num_classes * B, N)
        _r_batch = _r_batch.unsqueeze(0).expand(self.num_classes, -1, -1, -1).reshape(self.num_classes * B, M, 1)
        
        sub_D_batch = torch.bmm(_D_batch, Q_prod_batch)
        
        fid_batch = torch.sum(
            torch.square(_f_batch - torch.bmm(sub_D_batch, _r_batch).squeeze()), dim=1
        )
        fid_batch = torch.stack(torch.split(fid_batch.squeeze(), B)).T
        
        return -fid_batch
