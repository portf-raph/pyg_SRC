import torch
from torch.nn.parameter import Parameter

from utils.dict_learning import get_dict, scode_obj, FISTA

class SparseCoder(torch.nn.Module):
    def __init__(self,
                 K: int,
                 in_channels: int,  # C
                 num_atoms: int,  # M
                 num_classes: int,
                 _lambda: float,
                 _eta: float,
                 backward: bool,
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
        self._eta = _eta
        self.backward = backward
        self.device = device

        self.A = Parameter(torch.randn(self.K, self.in_channels*self.num_atoms)).to(device)   # TODO: A init
        self.A_loss = 0

        # Selection operators
        assert self.partition[0] == 0
        assert self.partition[-1] == num_atoms
        assert self.num_classes + 2 == len(self.partition)
        self.Q = torch.eye(self.num_atoms, device=device)

    def forward(self,
                data_dicts: list[dict]):
        self.A_loss = 0
        _r_batch = []
        for data_dict in data_dicts:       # TODO: parallel implementation
            edge_index = data_dict['edge_index']
            edge_attr = data_dict['edge_attr']
            x = data_dict['x']    # now GIN output
            y = data_dict['y']
            eigs = atol_eigs(data_dict['eigs'], edge_index)
            V = data_dict['V']

            D = get_dict(A=self.A,
                         K=self.K,
                         in_channels=self.in_channels,
                         V=V,
                         eigs=eigs)
            start = self.partition[y]
            end = self.partition[y+1]
            # free memory asap, so compute incoherence now
            if self.backward:
                _D_label = D[:, start:end]
                _D_rest = torch.cat((D[:, 0:start], D[:, end:self.partition[-1]]), dim=1)
                incoherence = torch.sum(torch.square(_D_label.T @ _D_rest))

            sub_Q = torch.cat((self.Q[:, start:end],
                               self.Q[:, self.partition[-2]:self.partition[-1]]), dim=1)
            sub_D = D @ (sub_Q @ sub_Q.T)
            Q_ex_T = torch.cat((self.Q[:, 0:start],
                                self.Q[:, end:self.partition[-2]]), dim=1).T
            padding = Q_ex_T.shape[0]
            _D_stack = torch.cat([D, sub_D, Q_ex_T], dim=0)
            f = x.t().flatten().view(-1,1)
            _f_stack = torch.cat(
                (f, f, torch.zeros((padding,1), device=self.device)),
                dim=0
                )
            _r = FISTA(_f_stack=_f_stack, _D_stack=_D_stack, _lambda=self._lambda)

            if self.backward:
                _D = _D_stack[0:-padding, :]
                _f = _f_stack.detach()[0:-padding, :]   # detach _f and _r from A_loss.backward()
                fidelity = (1/2) * torch.sum(torch.square(_f - _D @ _r.detach()))
                self.A_loss += fidelity + self._eta * incoherence
            _r_batch.append(_r)

        self.A_loss = self.A_loss / len(data_dicts)
        return _r_batch
