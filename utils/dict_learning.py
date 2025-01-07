import torch
from torch import Tensor
import torch.nn.functional as F
import torchopt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dict(A: Tensor,
             V: Tensor,
             eigs: Tensor,
             K: int,
             in_channels: int,
             ) -> Tensor:

    assert (A.shape[1] % in_channels == 0)

    G_k = []
    for k in range(K):
        G_k.append(torch.pow(eigs, k))
    G_k = torch.stack(G_k).T
    D = G_k @ A
    D = V @ D

    num_nodes = V.shape[0]
    D = D.t().flatten()
    D = D.split(num_nodes * in_channels, dim=0)
    D = torch.stack(D, dim=0)
    D = D.t()
    D = F.normalize(D,p=2, dim=0, eps=1e-12)

    return D


def scode_obj(params: Tensor,
                _f_stack: Tensor,
                _D_stack: Tensor,
                _lambda: float
                ):
    regress = (1/2) * torch.sum(torch.square(_f_stack - _D_stack @ params))
    penalty = _lambda * torch.linalg.norm(params, ord=1)
    return regress + penalty


def FISTA(
    _f_stack: torch.Tensor,
    _D_stack: torch.Tensor,
    _lambda: float
) -> torch.Tensor:

    c = PowerMethod(_D_stack)
    eta = 1 / c
    FISTA_ITER = 100

    params = soft_threshold(eta * _D_stack.T @ _f_stack, _lambda)
    Z = params.clone()
    t = 1

    for _ in range(FISTA_ITER):
        _r_1 = params.clone()
        residual = _D_stack @ Z - _f_stack
        params = soft_threshold(Z - eta * _D_stack.T @ residual, _lambda / c)

        t_1 = t
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        Z = params + ((t_1 - 1) / t * (params - _r_1)).to(_f_stack.device)

    return params


def PowerMethod(_D_stack):
    ITER = 100
    r = torch.randn(_D_stack.shape[1]).to(device)
    for i in range(ITER):
        Dr = _D_stack @ r
        r = _D_stack.T @ Dr
        nm = torch.norm(r,p=2)
        r = r/nm

    return nm


def soft_threshold(_r, _lambda):
    r = _r.clone()
    r = torch.sign(r) * F.relu(torch.abs(r)-_lambda)
    return r.to(device)
