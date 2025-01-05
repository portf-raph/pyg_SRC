import torch
from torch.nn.parameter import Parameter

from utils.dict_learning import get_dict, scode_obj, FISTA


class SparseCoder(torch.nn.Module):
    def __init__(self,
                 K: int,
                 in_channels: int,
                 num_atoms: int,
                 num_classes: int,
                 _lambda: float,
                 partition: list[int],
                 device='cpu'
                 ):

        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.num_atoms = num_atoms
        self.num_classes = num_classes
        self.partition = partition
        self._lambda = _lambda

        # Parameter A
        self.A = Parameter(torch.randn(self.K, self.in_channels*self.num_atoms)).to(device)   # TODO: A init

        # Selection operators
        assert self.partition[0] == 0
        assert self.partition[-1] == num_atoms + 1
        assert self.num_classes + 2 == len(self.partition)
        self.Q = torch.eye(self.num_atoms)

    def forward(self,
                data_dicts: list[dict]):
        _r_batch = []
        for data_dict in data_dicts:       # TODO: parallel implementation
            edge_index = data_dict['edge_index']
            edge_attr = data_dict['edge_attr']
            x = data_dict['x']    # now GIN output
            y = data_dict['y']
            eigs = atol_eigs(data_dict['eigs'], edge_index)
            V = torch.from_numpy(data_dict['V'])

            D = get_dict(A=self.A,
                         K=self.K,
                         in_channels=self.in_channels,
                         V=V,
                         eigs=eigs)
            start = self.partition[y]
            end = self.partition[y+1]
            sub_Q = torch.cat((self.Q[:, start:end],
                               self.Q[:, self.partition[-2]:self.partition[-1]]), dim=1)
            sub_D = D @ (sub_Q @ sub_Q.T)
            Q_ex = torch.cat((self.Q[:, 0:start],
                              self.Q[:, end:self.partition[-2]]), dim=1)

            _D_stack = torch.cat([D, sub_D, Q_ex.T], dim=0)
            f = x.t().flatten().view(-1,1)
            _f_stack = torch.cat(
                (f, f, torch.zeros((Q_ex.T.shape[0],1))),
                dim=0
                )
            _r = FISTA(_f_stack=_f_stack, _D_stack=_D_stack, _lambda=self._lambda)
            _r_batch.append(_r)

        return _r_batch
