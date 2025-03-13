import torch

from utils.data_helper import pyg_batch
from utils.network import MLP
from .GIN import GIN_Processor
from .scode_DLCOPAR import DL_COPAR
from .scode_DLCOPAR_vanilla import DL_COPAR_van
from .LeastEnergy import LeastEnergy


class SRCNet(torch.nn.Module):
    def __init__(self,
                 GIN_cfg: dict,
                 SC_cfg: dict,
                 OUT_cfg: dict,
                 model_SC: str,
                 model_class: str,
                 device='cpu'
                 ):
        super().__init__()
        self.device = device
        self.GIN = GIN_Processor(**GIN_cfg,
                                 device=device)
        self.model_SC = model_SC
        if model_SC = "DL_COPAR":
            self.SC = DL_COPAR(**SC_cfg,
                               device=device)
        elif model_SC = "DL_COPAR_van":
            self.SC = DL_COPAR_van(**SC_cfg,
                                   device=device)
        else:
            raise Exception("Invalid sparse coder")

        self.model_class = model_class
        if model_class == "MLP":
            self.OUT = MLP(**OUT_cfg,
                           device=device)
        elif model_class == "LeastEnergy":
            self.OUT = LeastEnergy(**OUT_cfg,
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
            
        del _D_batch, _f_batch
        return out, A_fidelity, A_incoherence
