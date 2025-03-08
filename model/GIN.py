import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from utils.network import MLP


class GIN_Processor(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 MLP_cfg: dict,
                 skip_first_features: bool=False,
                 device='cpu'):
        super().__init__()
        self.skip_first_features = skip_first_features
        self.convs = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers

        for layer in range(num_layers):
            if layer == 0:
                local_in_channels = in_channels
            else:
                local_in_channels = MLP_cfg['hid_dim']    # TODO: validate
            MLP_ = MLP(in_dim=local_in_channels,
                      out_dim=MLP_cfg['hid_dim'],
                      hid_dim=MLP_cfg['hid_dim'],
                      num_hid=MLP_cfg['num_hid'],
                      dp_cfg=MLP_cfg['dp_cfg'],
                      bn_cfg=MLP_cfg['bn_cfg'],
                      output_activation='relu', device=device)  # mod call
            GIN_layer = GINConv(MLP_, eps=0., train_eps=False).to(device)
            self.convs.append(GIN_layer)

    def forward(self, x, edge_index):

        x_cat = []
        if not self.skip_first_features:
            x_cat.append(x)

        x = self.convs[0](x, edge_index)
        x_cat.append(x)

        for layer in range(1, self.num_layers):
            x = self.convs[layer](x, edge_index)
            x_cat.append(x)

        x = torch.cat(x_cat, dim=1)
        return x
