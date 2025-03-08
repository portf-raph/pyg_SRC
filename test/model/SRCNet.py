import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset

from model.SRCNet import SRCNet
from data.pth_dataset import PthDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_SRCNet():
    GIN_cfg = {
        'in_channels': 3,
        'num_layers': 2,
        'skip_first_features': True,
        'MLP_cfg': {'hid_dim': 12, 'num_hid': 2,
                 'dp_cfg': {'skip_first': False, 'dropout': 0.0},
                 'bn_cfg': {'use_batchnorm': True, 'batchnorm_affine': True},
                 'output_activation': 'relu'
                 }
    }
    
    GIN_out_channels = GIN_cfg['num_layers'] * GIN_cfg['MLP_cfg']['hid_dim']
    SC_cfg = {
        'K': 10,
        'num_atoms': 30,
        'num_classes': 2,
        'in_channels': GIN_out_channels,
        'split_factor': 1,
        '_lambda': 0.1,
        '_eta': 0.3,
        'partition': [0,10,20,30],
    }
    in_dim = SC_cfg['num_atoms']
    out_dim = SC_cfg['num_classes']
    MLP_cfg = {
        'in_dim': in_dim,
        'out_dim': out_dim,
        'hid_dim': 64,
        'num_hid': 2,
        'dp_cfg': {'skip_first': False, 'dropout': 0.5},
        'bn_cfg': {'use_batchnorm': False, 'batchnorm_affine': True},
        'output_activation': 'linear'
    }
    LE_cfg = {
        'num_atoms': SC_cfg['num_atoms'],
        'num_classes': SC_cfg['num_classes'],
        'partition': SC_cfg['partition']
    }
    model = SRCNet(GIN_cfg, SC_cfg, LE_cfg, model_class="LeastEnergy", device=device)
    params = filter(lambda p: p.requires_grad and p is not model.SC.A,
                        model.parameters())
    params = list(params)
    A = model.SC.A
    optimizer = torch.optim.Adam([
        {'params': params,
        'lr': 1e-4,
        'weight_decay': 0.1},
        {'params': A,
        'lr': 1e-4,
        'weight_decay': 0.05}
    ])
    optimizer.zero_grad()

    dataset = PthDataset(load_dir='./data/PROTEINS/pth/train')
    data_dicts = [dataset[i] for i in range(8)]
    data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                    else v for k, v in d.items()} for d in data_dicts]
    y = torch.tensor([d['y'].item() for d in data_dicts], dtype=torch.long, device=device)
    
    #model.eval()
    #out, _, _ = model(data_dicts)
    
    out, A_fidelity, A_incoherence = model(data_dicts)
    A_loss = A_fidelity + model.SC._eta * A_incoherence
    A_loss.backward(retain_graph=True)
    A_grad = model.SC.A.grad.detach().clone()
    
    for p in list(model.GIN.parameters()):
        assert p.grad == None

    train_loss = F.cross_entropy(out, y)
    torch.autograd.backward(train_loss, inputs=params)
    
    assert train_loss < 2.0
    for p in list(model.GIN.parameters()):
        assert p.grad != None
    assert torch.allclose(model.SC.A.grad, A_grad)
    
    optimizer.step

    return 0
