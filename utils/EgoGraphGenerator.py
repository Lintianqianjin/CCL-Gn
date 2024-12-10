import sys

from tqdm import tqdm

import torch
import torch.nn.functional as F

# from torch.utils.data.distributed import DistributedSampler

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Dict, Callable

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Batch, Data
# from torch_geometric.data import Data
# from torch_geometric.loader.utils import edge_type_to_str
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler
from torch_geometric.sampler.utils import to_csc
from torch_geometric.loader.utils import filter_data
from torch_sparse import SparseTensor, matmul

from copy import deepcopy

# Start ####################### Full Ego Graph Dataset ####################### Start #
# class PYGEgoGraphDataset(torch.utils.data.Dataset):
    
#     def __init__(self, pyg, num_hops, idxes, de=False):
#         'Initialization'
#         self.pyg = pyg
#         self.idxes = idxes
#         self.de = de
        
#         attrs = pyg.to_dict().keys()
        
#         node_attrs = list(filter(lambda x: pyg.is_node_attr(x), attrs))
#         self.node_attrs = node_attrs
        
#         edge_attrs = list(filter(lambda x: pyg.is_edge_attr(x) and x != 'edge_index', attrs))
#         self.edge_attrs = edge_attrs
        
#         self.num_hops = num_hops
        
#     def __len__(self):
#         'Denotes the total number of samples'
#         return self.idxes.shape[0]

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         # %timeit 100
#         index = self.idxes[index]
#         # print(type(index))
#         if isinstance(index, torch.Tensor):
#             index = index.item()

#         tmp_ego_graph = k_hop_subgraph(
#             node_idx=index, num_hops=self.num_hops, edge_index=self.pyg.edge_index, 
#             relabel_nodes=True, num_nodes=None, flow='source_to_target'
#         )
#         node_idxes, new_egde_idxes, center_new_idx, edge_mask = tmp_ego_graph[0], tmp_ego_graph[1], tmp_ego_graph[2], tmp_ego_graph[3]
#         center_mask = F.one_hot(center_new_idx, num_classes=node_idxes.shape[0]).reshape(node_idxes.shape[0],1)
        
#         node_kvs = {node_attr: self.pyg[node_attr][node_idxes] for node_attr in self.node_attrs}
#         node_kvs['center_mask'] = center_mask
#         edge_kvs = {edge_attr: self.pyg[edge_attr][edge_mask] for edge_attr in self.edge_attrs}
#         z = {**node_kvs, **edge_kvs}
#         z['edge_index'] = new_egde_idxes
        
#         sample_data = Data(**z)

#         return sample_data

# End ####################### Full Ego Graph Dataset ####################### End #
import numpy as np
from scipy.sparse import csc_matrix, diags

#
def get_features_rw_sample(edge_index, center_mask, rw_depth):
    center_idx = np.argmax(center_mask.numpy())
    num_nodes = center_mask.shape[0]
    epsilon = 1e-6

    adj = csc_matrix((np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())),
                     shape = (num_nodes, num_nodes)
          )
    # adj = SparseTensor.from_edge_index(
    #         edge_index = edge_index,
    #         sparse_sizes = (num_nodes, num_nodes)
    # )

    out_d = (adj.sum(1) + epsilon)
    out_d = np.asarray(out_d).flatten()

    D = diags(1/out_d)
   
    adj = D @ adj
    # print(adj.sum(1) + epsilon)

    rw_list = [adj]
    for _ in range(1, rw_depth):
        rw = rw_list[-1]@adj
        rw_list.append(rw)
   
    rw_feat = np.concatenate([rw[:, center_idx].todense() for rw in rw_list], axis=-1)

    return torch.tensor(rw_feat).float()
#

# Start ####################### Sample Ego Graph Dataset ####################### Start #

NumNeighbors = List[int]

class NeighborSampler:
    def __init__(
        self,
        data: Data,
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed

        if isinstance(data, Data):
            # Convert the graph data into a suitable format for sampling.
            self.colptr, self.row, self.perm = to_csc(data)
            assert isinstance(num_neighbors, (list, tuple))


    def __call__(self, indices: List[int]):
        if not isinstance(indices, torch.Tensor):
            index = torch.tensor(indices)
        assert index.dtype == torch.int64

        if isinstance(self.data, Data):
            sample_fn = torch.ops.torch_sparse.neighbor_sample
            node, row, col, edge = sample_fn(
                self.colptr,
                self.row,
                index,
                self.num_neighbors,
                self.replace,
                self.directed,
            )
            data = filter_data(self.data, node, row, col, edge, self.perm)
            # data.batch_size = index.numel()

        return data

class PYGEgoSampledGraphDataset(torch.utils.data.Dataset):
    
    def __init__(self, pyg, num_hops, idxes, num_neighbors, rw_feat:bool, cache:bool):
        'Initialization'
        self.pyg = deepcopy(pyg)
        self.idxes = idxes
        self.rw_feat = rw_feat
        self.pyg['nid'] = torch.arange(self.pyg.x.shape[0])
        # self.pyg['eid'] = torch.arange(self.pyg.edge_index.shape[1])

        self.cache = cache
        # self.device = device
        
        self.attrs = self.pyg.to_dict().keys()
        
        self.node_attrs = list(filter(lambda x: self.pyg.is_node_attr(x), self.attrs))
        self.node_attrs = {node_attr: self.pyg[node_attr] for node_attr in self.node_attrs if node_attr != 'nid'}
        
        self.edge_attrs = list(filter(lambda x: self.pyg.is_edge_attr(x) and x != 'edge_index', self.attrs))
        self.edge_attrs = {edge_attr: self.pyg[edge_attr] for edge_attr in self.edge_attrs}

        for na in self.node_attrs.keys():
            # if na != 'nid': 
            del self.pyg[na]
        
        self.num_hops = num_hops
        self.num_neighbors = num_neighbors
        
        # sample_fn
        self.sample_fn = NeighborSampler(
            data = self.pyg,
            num_neighbors = self.num_neighbors,
            replace = False,
            directed = False,   
        )
        if self.cache:
            self.cache_list = [None] * self.idxes.shape[0]
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.idxes.shape[0]
    
    
    def __getitem__(self, index):
        # print(index)
        raw_index = index
        if not self.cache or self.cache_list[index] is None:
            'Generates one sample of data'
            # Select sample
            index = self.idxes[index] # node id in the pyg
            # print(index)
            # print(type(index))
            if isinstance(index, torch.Tensor):
                index = index.item()
            # print(index)

            sample_data =  self.sample_fn([index])
            center_mask = (sample_data.nid == index).long()
            sample_data['center_mask'] = center_mask

            if self.rw_feat:
                rw_feat = get_features_rw_sample(sample_data.edge_index, center_mask, rw_depth=self.num_hops)
                sample_data['rw_feat'] = rw_feat

            sample_data = sample_data #.to(self.device)

            if self.cache:
                self.cache_list[raw_index] = sample_data
            # return sample_data
        else:
            sample_data = self.cache_list[index]
        
        complete_data = deepcopy(sample_data)
        # assign node feature
        for node_attr_name, node_attr_tensor in self.node_attrs.items():
            # if node_attr_name != 'nid':
            complete_data[node_attr_name] = node_attr_tensor[sample_data.nid]
        # # print(f"{sample_data=}")

        return complete_data


def get_tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()

def get_data_object_size(data):
    size_bytes = 0
    for key, item in data:
        if torch.is_tensor(item):
            size_bytes += get_tensor_size(item)
        elif isinstance(item, int) or isinstance(item, float):
            size_bytes += sys.getsizeof(item)
        # You can add more checks here if you have other types in your Data object
    return size_bytes  / (1024 * 1024)


# End ####################### Sample Ego Graph Dataset ####################### End #
    
# def PYGEgoGraphDataLoader(pyg, node_idxes, num_hops, batch_size, shuffle, drop_last, num_workers, pin_memory):
#     '''
#     '''
#     dataset = PYGEgoGraphDataset(pyg=pyg, num_hops=num_hops, idxes=node_idxes)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=shuffle,
#         num_workers=num_workers, collate_fn=Batch.from_data_list, 
#         pin_memory=pin_memory, drop_last=drop_last, timeout=0, worker_init_fn=None, 
#         multiprocessing_context=None, generator=None, sampler=None
#     )
    
#     return dataloader

# def PYGDistEgoGraphDataLoader(pyg, node_idxes, num_hops, batch_size, shuffle, drop_last, num_workers, pin_memory):
#     dataset = PYGEgoGraphDataset(pyg=pyg, num_hops=num_hops, idxes=node_idxes)

#     sampler = DistributedSampler(dataset, drop_last=drop_last, shuffle=shuffle)

#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=None,
#         num_workers=num_workers, collate_fn=Batch.from_data_list, 
#         pin_memory=pin_memory, drop_last=drop_last, timeout=0, worker_init_fn=None, 
#         multiprocessing_context=None, generator=None, sampler=sampler
#     )
#     return dataloader, sampler


# def PYGEgoSampledGraphDataLoader(pyg, node_idxes, num_hops, num_neighbors, batch_size, shuffle, drop_last, num_workers, pin_memory):
#     '''
#     '''
#     dataset = PYGEgoSampledGraphDataset(pyg=pyg, num_hops=num_hops, idxes=node_idxes, num_neighbors=num_neighbors)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=shuffle,
#         num_workers=num_workers, collate_fn=Batch.from_data_list, 
#         pin_memory=pin_memory, drop_last=drop_last, timeout=0, worker_init_fn=None, 
#         multiprocessing_context=None, generator=None, sampler=None
#     )
    
#     return dataloader

# def PYGDistEgoSampledGraphDataLoader(pyg, node_idxes, num_hops, num_neighbors, batch_size, shuffle, drop_last, num_workers, pin_memory):
#     dataset = PYGEgoSampledGraphDataset(pyg=pyg, num_hops=num_hops, idxes=node_idxes, num_neighbors=num_neighbors)

#     sampler = DistributedSampler(dataset, drop_last=drop_last, shuffle=shuffle)

#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=None,
#         num_workers=num_workers, collate_fn=Batch.from_data_list, 
#         pin_memory=pin_memory, drop_last=drop_last, timeout=0, worker_init_fn=None, 
#         multiprocessing_context=None, generator=None, sampler=sampler
#     )
#     return dataloader, sampler