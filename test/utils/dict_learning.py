import time
import torch

from ...utils.random_graph import er_graph
from ...utils.dict_learning import get_dict, get_R, get_OmegaP


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
    D = get_dict(A=A, V=V, eigs=eigs, K=K, in_channels=C)   # mod call
    et = time.time()

    assert D.shape[0] == 5*C, print('Wrong row shape')
    assert D.shape[1] == M, print('Wrong column shape')

    print('{}s elapsed'.format(et-st))
    return 0


def test_get_R():
    V = torch.ones(5,5)
    eigs = torch.ones(5)
    _r = torch.arange(0,4)
    K=5
    # VG_k == 5 * torch.ones(5,5)

    validR_r = [torch.zeros(5,5)]
    for i in range(1,4):
        validR_r.append(i*5*torch.ones(5,5))

    validR_r = torch.cat(validR_r, dim=1)
    R_r = get_R(V, eigs, _r, K)
    assert torch.allclose(R_r, validR_r)
    return 0


def test_get_OmegaP():
    num_classes = 2
    num_bins = 5
    num_atoms = 6
    K = 3
    class_weights = [(i+1)*torch.ones(num_bins, dtype=torch.float) for i in range(num_classes)]
    #print('class_weights: {}'.format(class_weights))
    bin_centers = torch.arange(num_bins, dtype=torch.float)
    partition = [0,3,6,9]
    class_idx = [0,0,0,1,1,1]

    P = torch.stack([torch.pow(bin_centers,k) for k in range(K)]).T
    #print('P: {}'.format(P))
    validOmegaP = []
    for i in class_idx:
        prod = torch.diag(class_weights[i]) @ P
        validOmegaP.append(prod)
    validOmegaP = torch.cat(validOmegaP, dim=1)

    class_weights = torch.stack(class_weights)
    OmegaP = get_OmegaP(class_weights=class_weights,
                        bin_centers=bin_centers,
                        partition=partition,
                        K=K)

    assert torch.allclose(OmegaP, validOmegaP)
    return 0


def test_f_split_mend():
    torch.manual_seed(1)
    batch_size = 3
    num_nodes_base = 2
    in_channels = 3
    padding = 2
    split_factor = 2
    _f_stack = [torch.randint(5, (num_nodes_base*i,in_channels)) for i in range(1,batch_size+1)]
    _f_split = build_f_split(split_factor)
    _f_mend = build_f_mend(in_channels=in_channels,
                             split_factor=split_factor,
                             padding=padding)
    print("Initial _f stack, shape: {}, {}".format(_f_stack, [tensor.shape for tensor in _f_stack]))
    _f_stack = _f_split(_f_stack)
    print("Split: {}, shape: {}".format(_f_stack, [tensor.shape for tensor in _f_stack]))
    _f_stack = _f_mend(_f_stack, batch_size)
    print("Mend: {}, shape: {}".format(_f_stack, [tensor.shape for tensor in _f_stack]))
