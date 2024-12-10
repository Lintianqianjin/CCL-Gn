import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from torch_geometric.loader import NeighborSampler
# from torch_geometric.nn import SAGEConv

from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: Tensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size) # out was l2 normalized
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            x_r = F.normalize(x_r, p=2., dim=-1)
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


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

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

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
        self.in_dim = config.in_dim
        self.hid_dim = config.hid_dim
        self.num_layers = config.num_layers

        self.convs = torch.nn.ModuleList()
        # self.bns = torch.nn.ModuleList()

        self.convs.append(SAGEConv(self.in_dim, self.hid_dim, normalize = False, root_weight = True))
        # self.bns.append(torch.nn.BatchNorm1d(self.hid_dim))

        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hid_dim, self.hid_dim, normalize = False, root_weight = True))
            # self.bns.append(torch.nn.BatchNorm1d(self.hid_dim))

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
                x = conv(x, edge_index, edge_weight=edge_weight)
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