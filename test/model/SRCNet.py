import torch
from ...model.SRCNet import SRCNet

from torch_geometric.datasets import TUDataset

def test_SRCNet():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    GIN_cfg = {
        'in_channels': 3,
        'num_layers': 2,
        'skip_first_features': True,
        'MLP_cfg': {'hid_dim': 10, 'num_hid': 2,
                 'dp_cfg': {'skip_first': False, 'dropout': 0.0},
                 'bn_cfg': {'use_batchnorm': True, 'batchnorm_affine': True},
                 'output_activation': 'relu'
                 }
    }
    SC_cfg = {
        'K': 10,
        'num_atoms': 30,
        'num_classes': 2,
        '_lambda': 0.2,
        '_eta': 0.3,
        'backward': True,
        'partition': [0,10,20,30]
    }
    OUT_cfg = {
        'hid_dim': 64,
        'num_hid': 2,
        'dp_cfg': {'skip_first': False, 'dropout': 0.5},
        'bn_cfg': {'use_batchnorm': False, 'batchnorm_affine': True},
        'output_activation': 'linear'
    }
    model = SRCNet(GIN_cfg, SC_cfg, OUT_cfg, device=device)

    dataset = TUDataset(root='../data', name='PROTEINS')
    data_dicts = []
    for i in range(5):
        data = dataset[i]

        data_dict = {}
        data_dict['edge_index'] = data.edge_index
        data_dict['edge_attr'] = data.edge_attr
        data_dict['x'] = data.x
        data_dict['y'] = data.y

        dense_adj = to_dense_adj(edge_index=data.edge_index,
                                 edge_attr=data.edge_attr)
        eigs, V = get_eigs(dense_adj)
        data_dict['eigs'] = eigs
        data_dict['V'] = V

        data_dicts.append(data_dict)

    for params in model.parameters():
        print('p shape:{}, p device: {}'.format(params.shape, params.device))
    params = filter(lambda p: p.requires_grad and p is not model.SC.A,
                        model.parameters())
    params = list(params)
    A = model.SC.A
    # forward
    optimizer = torch.optim.Adam([
        {'params': params,
        'lr': 1e-3,
        'weight_decay': 0.1},
        {'params': A,
        'lr': 1e-4,
        'weight_decay': 0.05}
    ])
    optimizer.zero_grad()

    data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                    else v for k, v in d.items()} for d in data_dicts]
    y = torch.LongTensor([d['y'].item() for d in data_dicts]).to(device)
    out = model(data_dicts)

    for gp in list(model.GIN.parameters()):
        print(gp.grad)
    print('==== none ====')
    # backward
    model.SC.A_loss.backward(retain_graph=True)
    # A_loss backward step shouldn't propagate to _f nor _r, and by extension GIN parameters.
    for p in list(model.GIN.parameters()):
        print(p.grad)

    train_loss = F.cross_entropy(out, y)
    torch.autograd.backward(train_loss,
                            inputs=list(params))
    for p in list(model.GIN.parameters()):
        print(p.grad)
    optimizer.step()

    return 0
