import torch

from utils.data_helper import pyg_batch
from utils.network import MLP
from .GIN import GIN_Processor
from .SparseCoder import SparseCoder
from .LeastEnergy import LeastEnergy
from .LeastActivation import LeastActivation


class SRCNet(torch.nn.Module):
    def __init__(self,
                 GIN_cfg: dict,
                 SC_cfg: dict,
                 OUT_cfg: dict,
                 model_class: str,
                 device='cpu'
                 ):
        super().__init__()
        self.device = device
        self.GIN = GIN_Processor(**GIN_cfg,
                                 device=device)
        self.SC = SparseCoder(**SC_cfg,
                              device=device)
        self.model_class = model_class
        
        if model_class == "MLP":
            self.OUT = MLP(**OUT_cfg,
                           device=device)
        elif model_class == "LeastEnergy":
            self.OUT = LeastEnergy(**OUT_cfg,
                                   device=device)
        elif model_class == "LeastActivation":
            self.OUT = LeastActivation(**OUT_cfg,
                                       device=device)
        else:
            raise Exception("Invalid classification model")

    def forward(self,
                data_dicts: list[dict]):
        batch = pyg_batch(data_dicts)
        batch.x = self.GIN(batch.x, batch.edge_index)
        batch_vec = batch.x_batch
        for i in range(len(data_dicts)):
            # assert data_dicts[i]['x'].shape[0] == torch.sum(batch_vec==i)
            data_dicts[i]['x'] = batch.x[batch_vec==i]

        _r_batch, A_fidelity, A_incoherence, _D_batch, _f_batch = self.SC(data_dicts)
        
        if self.model_class == "MLP":
            out = self.OUT(_r_batch.squeeze())
            
        elif self.model_class == "LeastEnergy":
            out = self.OUT(_r_batch=_r_batch,
                           _D_batch=_D_batch,
                           _f_batch=_f_batch)
        elif self.model_class == "LeastActivation":
            out = self.OUT(_r_batch.squeeze())

        del _D_batch, _f_batch
        return out, A_fidelity, A_incoherence
