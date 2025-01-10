from utils.network import MLP
from model.GIN import GIN_Processor
from model.SparseCoder import SparseCoder

import torch


import torch

class SRCNet(torch.nn.Module):
    def __init__(self,
                 GIN_cfg: dict,
                 SC_cfg: dict,
                 OUT_cfg: dict,
                 device='cpu'
                 ):
        super().__init__()
        self.device = device
        self.GIN = GIN_Processor(**GIN_cfg,
                                 device=device)
        self.SC = SparseCoder(**SC_cfg,
                       in_channels=self.GIN.out_channels,
                       device=device)
        self.OUT = MLP(**OUT_cfg,
                       in_dim=self.SC.num_atoms,
                       out_dim=self.SC.num_classes,
                       device=device)

    def forward(self,
                data_dicts: list[dict]):
        # GIN input: x, edge_index
        batch = pyg_batch(data_dicts)
        batch.x = self.GIN(batch.x, batch.edge_index)
        batch_vec = batch.x_batch
        for i in range(len(data_dicts)):
            assert data_dicts[i]['x'].shape[0] == torch.sum(batch_vec==i)
            data_dicts[i]['x'] = batch.x[batch_vec==i]

        out = self.SC(data_dicts)
        out = torch.squeeze(torch.stack(out))
        out = self.OUT(out)
        return out
