import torch
from torch import Tensor

class LeastActivation(torch.nn.Module):
    def __init__(self,
                 num_classes: int,
                 partition: list[int],
                 device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.partition = partition
        assert self.partition[0] == 0
        assert self.num_classes + 2 == len(self.partition)

    def forward(self,
                _r_batch: Tensor):
        
        _r_batch = torch.split(_r_batch, self.partition[1], dim=1)
        _r_batch = torch.stack([_r_batch[i] for i in range(self.num_classes)])
        _r_batch = torch.abs(_r_batch)
        _r_hot = torch.linalg.norm(_r_batch, dim=-1).squeeze()
        return _r_hot.T
