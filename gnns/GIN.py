import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

from typing import Union, Tuple, Callable, Optional
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
import torch.nn as nn
from torch.nn import Sequential, BatchNorm1d, ReLU, Tanh
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.nn import GINConv


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        # self.bound = torch.nn.Tanh()
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        # self.nn.reset_parameters()
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.nn.apply(weight_reset)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: Tensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight = edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            x_r = F.normalize(x_r, p=2., dim=-1)
            out += (1 + self.eps) * x_r

        # DIY bound
        # out = self.bound(out)

        return self.nn(out)


    def message(self, x_j: Tensor, edge_weight: Tensor=None) -> Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1,1)*x_j
        
    def update(self, inputs):
        return F.normalize(inputs, p=2., dim=-1)

    # def message_and_aggregate(self, adj_t: SparseTensor,
    #                           x: OptPairTensor) -> Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class Config(object):
    def __init__(self, **kwargs):
        '''
        '''
        self.in_dim = 128
        self.hid_dim = 128
        self.num_layers = 2

class Net(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config.in_dim
        self.dim = config.hid_dim
        self.out_channels = config.hid_dim
        self.num_layers = config.num_layers

        self.convs = torch.nn.ModuleList()
        # self.bns = torch.nn.ModuleList()

        self.convs.append(GINConv(
            Sequential(Linear(self.in_channels, self.dim), BatchNorm1d(self.dim), ReLU(),
                       Linear(self.dim, self.dim))))
        # self.bns.append(torch.nn.BatchNorm1d(self.dim))

        for _ in range(self.num_layers - 1):
            self.convs.append(GINConv(
                                Sequential(Linear(self.dim, self.dim), BatchNorm1d(self.dim), ReLU(),
                                        Linear(self.dim, self.dim))))
            # self.bns.append(torch.nn.BatchNorm1d(self.dim))

        self.activation = nn.ReLU(inplace=False)
        self.reset_parameters()
    
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            # x = self.bns[i](x)
            if i != self.num_layers - 1:
                x = self.activation(x)
                # x = F.dropout(x, p=0.5, training=self.training)
        return x

    # @torch.no_grad()
    # def inference(self, dataloader, x_all):
    #     pbar = tqdm(total=x_all.size(0) * self.num_layers)
    #     pbar.set_description('Evaluating')

    #     # Compute representations of nodes layer by layer, using *all*
    #     # available edges. This leads to faster computation in contrast to
    #     # immediately computing the final representations of each batch.
    #     # total_edges = 0
    #     for i in range(self.num_layers):
    #         xs = []
    #         for batch_size, n_id, adj in dataloader:
    #             edge_index, _, size = adj.to(0)
    #             # total_edges += edge_index.size(1)
    #             x = x_all[n_id].to(0)
    #             x_target = x[:size[1]]
    #             x = self.convs[i]((x, x_target), edge_index)
    #             x = self.bns[i](x)
    #             if i != self.num_layers - 1:
    #                 x = self.activation(x)
    #             xs.append(x)

    #             pbar.update(batch_size)

    #         x_all = torch.cat(xs, dim=0)

    #     pbar.close()

    #     return x_all