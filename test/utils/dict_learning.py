import torch
import time

from ...utils.random_graph import er_graph
from ...utils.dict_learning import get_dict, scode_obj, FISTA

def test_get_dict(K):
    if K is None:
        K = 10
    C = 12
    M = 32
    torch.manual_seed(1)

    A = torch.randn(K, C*M)
    V = torch.randn(5,5)
    eigs = torch.randn(5)

    st = time.time()
    D = get_dict(A=A, V=V, eigs=eigs, K=K, in_channels=C)
    et = time.time()

    assert D.shape[0] == 5*C, print('Wrong row shape')
    assert D.shape[1] == M, print('Wrong column shape')

    print('{}s elapsed'.format(et-st))
    return 0


def test_FISTA():
    torch.manual_seed(1)
    _D_stack = torch.randn(45, 30)
    _lambda = .05

    # synthetic
    _r_syn = torch.cat([torch.zeros(3),
                        torch.Tensor([1,0,1]),
                        torch.zeros(24)])
    _f_stack = _D_stack @ _r_syn
    params = FISTA(_f_stack=_f_stack, _D_stack=_D_stack, _lambda=_lambda)
    grad = torch.func.grad(scode_obj, argnums=0)(params, _f_stack, _D_stack, _lambda)
    assert torch.allclose(params, _r_syn, atol=1e-2)
    assert torch.allclose(grad, torch.zeros(30), atol=5e-1)

    # random
    _f_stack_0 = torch.randn(45)
    _f_stack_0.requires_grad = True
    params = FISTA(_f_stack=_f_stack_0, _D_stack=_D_stack, _lambda=_lambda)
    grad = torch.func.grad(scode_obj, argnums=0)(params, _f_stack_0, _D_stack, _lambda)
    assert torch.allclose(grad, torch.zeros(30), atol=5e-1)

    return 0
