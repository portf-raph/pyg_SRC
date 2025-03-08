# Modified from https://github.com/jsulam/Online-Dictionary-Learning-demo
import numpy as np
from collections.abc import Callable

import torch
from torch import Tensor
import torch.nn.functional as F

import torchopt
from torchopt.typing import TupleOfTensors

from .dict_learning import scode_obj_bmm, pad_columns


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#@torchopt.diff.implicit.custom_root(
#    torch.func.grad(scode_obj_bmm, argnums=0),
#    argnums=1,
#    solve=torchopt.linear_solve.solve_normal_cg(maxiter=30, atol=1e-6),   # ISSUE #1: parallelize custom_root(), i.e. torchopt.linalg.cg
#)
def FISTA_bmm(
    params,   # preserve scode_obj signature
    _f_stack: TupleOfTensors,
    _D_stack: Tensor,
    _lambda: float,
    _f_mend: Callable,
) -> TupleOfTensors:

    _f_stack = _f_mend(_f_stack=_f_stack,
                       batch_size=_D_stack.shape[0]).unsqueeze(1).contiguous()

    W = torch.linalg.norm(_D_stack, ord=2, dim=1)
    _D = _D_stack / W.unsqueeze(1)   # UNIT
    c = PowerMethod_bmm(_D)   
    eta = 1 / c
    FISTA_ITER = 200

    params = soft_threshold(eta * torch.bmm(_f_stack, _D), _lambda)   
    Z = params.clone()
    t = 1

    for _ in range(FISTA_ITER):
        _r_1 = params.clone()
        residual = torch.bmm(Z, _D.transpose(-1,-2)) - _f_stack  
        params = soft_threshold(Z - eta * torch.bmm(residual, _D), _lambda / c)   

        t_1 = t
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        Z = params + ((t_1 - 1) / t * (params - _r_1))
    params = (params.squeeze() / W).unsqueeze(-1)   # RESCALE
    return torch.unbind(params)   


def PowerMethod_bmm(_D):
    ITER = 100
    r = torch.randn((_D.shape[0], _D.shape[2], 1), device=device)  
    for i in range(ITER):
        Dr = torch.bmm(_D, r)   
        r = torch.bmm(_D.transpose(-1,-2), Dr)   
        nm = torch.norm(r,p=2)
        r = r/nm

    return nm


def soft_threshold(_r, _lambda):
    r = _r.clone()
    r = torch.sign(r) * F.relu(torch.abs(r)-_lambda)
    return r
