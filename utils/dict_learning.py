import torch
from torch import Tensor
import torch.nn.functional as F


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

    W = torch.linalg.norm(_D_stack, ord=2, dim=0)
    _D = _D_stack / W.unsqueeze(0)   # UNIT; no need to detach b/c of use of torch.autograd.backward()
    c = PowerMethod(_D)
    eta = 1 / c
    FISTA_ITER = 200

    params = soft_threshold(eta * _D.T @ _f_stack, _lambda)
    Z = params.clone().to(device)
    t = 1

    for _ in range(FISTA_ITER):
        _r_1 = params.clone()
        residual = _D @ Z - _f_stack
        params = soft_threshold(Z - eta * _D.T @ residual, _lambda / c)

        t_1 = t
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        Z = params + ((t_1 - 1) / t * (params - _r_1))
    return params.squeeze() / W   # RESCALE


def PowerMethod(_D):
    ITER = 100
    r = torch.randn(_D.shape[1], device=device)
    for i in range(ITER):
        Dr = _D @ r
        r = _D.T @ Dr
        nm = torch.norm(r,p=2)
        r = r/nm

    return nm


def soft_threshold(_r, _lambda):
    r = _r.clone()
    r = torch.sign(r) * F.relu(torch.abs(r)-_lambda)
    return r.to(device)
