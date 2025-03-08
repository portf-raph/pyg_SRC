import math

import torch
import torch.nn.functional as F

from utils.FISTA import FISTA_bmm
from model.LeastEnergy import LeastEnergy


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_LeastEnergy():
    torch.manual_seed(0)
    N, m = 10, 5
    _lambda = 0.03
    
    _f_1 = torch.ones((N,), device=device)
    _f_2 = -5*torch.ones((N,), device=device)
    _D = F.pad(
        torch.stack([_f_1, _f_2],dim=1), pad=(0,m-2), value=1
    )
    
    _f_batch = torch.stack([_f_1, _f_2])
    _D_batch = torch.stack([_D, _D])
    _f_mend = lambda _f_stack, batch_size : _f_stack
    _r_batch = FISTA_bmm(None,
                         _f_stack=_f_batch,
                         _D_stack=_D_batch,
                         _lambda=_lambda,
                         _f_mend=_f_mend)

    fid_1 = torch.sum(torch.square(_f_1 - _D @ _r_batch[0].squeeze()))
    fid_2 = torch.sum(torch.square(_f_1 - _D @ _r_batch[1].squeeze()))  # intented wrong
    out_1 = torch.exp(-fid_1)
    out_2 = torch.exp(-fid_2)
    
    _r = torch.cat([_r_batch[0], _r_batch[1], torch.zeros((m,1), device=device)], dim=0)
    _D_batch = torch.unbind(_D_batch)
    _D_batch = torch.cat(_D_batch, dim=1)
    _D_batch = F.pad(_D_batch, pad=(0,m,0,0))
    LE = LeastEnergy(num_atoms=3*m,
                     num_classes=2,
                     partition=[0,5,10,15],
                     device=device)
    out = LE(_r_batch=_r.unsqueeze(0),
             _D_batch=_D_batch.unsqueeze(0),
             _f_batch=_f_1.unsqueeze(0))
    
    assert math.isclose(out[0][0].item(), out_1.item())
    assert math.isclose(out[0][1].item(), out_2.item())
  
    return 0
