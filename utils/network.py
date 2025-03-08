import torch


class MLP(torch.nn.Module):
    def __init__(self,
                 in_dim, out_dim, hid_dim, num_hid,
                 dp_cfg: dict, bn_cfg: dict,
                 output_activation='relu',
                 device='cpu'
                 ):
        super().__init__()
        self.num_hid = num_hid
        self.layers = torch.nn.ModuleList()
        self.dp_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.skip_first = dp_cfg['skip_first']

        if not self.skip_first:
            self.dp_layers.append(torch.nn.Dropout(dp_cfg['dropout']))

        if self.num_hid == 0:
            self.layers.append(torch.nn.Linear(in_dim, out_dim, device=device))
        elif self.num_hid > 0:
            self.layers.append(torch.nn.Linear(in_dim, hid_dim, device=device))
            if bn_cfg['use_batchnorm']:
                self.bn_layers.append(torch.nn.BatchNorm1d(hid_dim, affine=bn_cfg['batchnorm_affine'], device=device))
            else:
                self.bn_layers.append(torch.nn.Identity())

            for layer in range(num_hid - 1):
                self.layers.append(torch.nn.Linear(hid_dim, hid_dim, device=device))
                self.dp_layers.append(torch.nn.Dropout(dp_cfg['dropout']))
                if bn_cfg['use_batchnorm']:
                    self.bn_layers.append(torch.nn.BatchNorm1d(hid_dim, affine=bn_cfg['batchnorm_affine'], device=device))
                else:
                    self.bn_layers.append(torch.nn.Identity())

            self.layers.append(torch.nn.Linear(hid_dim, out_dim, device=device))
            self.dp_layers.append(torch.nn.Dropout(dp_cfg['dropout']))

        self.activation = torch.nn.functional.relu
        if output_activation == 'relu':
            self.output_activation = torch.nn.functional.relu
        elif output_activation == 'linear':
            self.output_activation = None
        else:
            raise 'unknown activation for output layer of MLP'

    def forward(self, x):
        if self.num_hid == 0:
            if not self.skip_first:
                x = self.dp_layers[0](x)
            x = self.layers[0](x)
            if not (self.output_activation is None):
                x = self.output_activation(x)
        else:
            if not self.skip_first:
                x = self.dp_layers[0](x)

            x = self.bn_layers[0](self.layers[0](x))
            x = self.activation(x)
            for layer in range(1, self.num_hid):
                x = self.dp_layers[layer](x)
                x = self.bn_layers[layer](self.layers[layer](x))
                x = self.activation(x)
            self.dp_layers[-1](x)
            x = self.layers[-1](x)
            if not (self.output_activation is None):
                x = self.output_activation(x)
        return x
