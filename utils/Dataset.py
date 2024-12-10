import os
import random
random.seed(0)

import pandas as pd
import numpy as np
np.random.seed(0)

import torch
torch.manual_seed(0)

import scipy

from tqdm import tqdm

from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.datasets import Flickr, Amazon, CitationFull
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from torch_geometric.nn import Node2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
from ogb.nodeproppred import NodePropPredDataset

from GOOD.data.good_datasets.good_webkb import GOODWebKB
from GOOD.data.good_datasets.good_twitch import GOODTwitch
from GOOD.data.good_datasets.good_cora import GOODCora
from GOOD.data.good_datasets.good_cbas import GOODCBAS
from GOOD.data.good_datasets.good_arxiv import GOODArxiv

import pickle as pkl

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_nc_dataset(data_dir, dataname, year=2020):
    """ Loader for NCDataset
        Returns NCDataset
    """
    if dataname == 'ogb-arxiv':
        dataset = load_ogb_arxiv(data_dir=data_dir, year_bound=year, proportion = 1.0)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def take_second(element):
    return element[1]

def load_ogb_arxiv(data_dir, year_bound = [2018, 2020], proportion = 1.0):
    import ogb.nodeproppred

    dataset = ogb.nodeproppred.NodePropPredDataset(name='ogbn-arxiv', root=data_dir)
    graph = dataset.graph

    node_years = graph['node_year']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years))

    edges = graph['edge_index']
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] <= year_bound[1] and node_years[edges[1][i]] <= year_bound[1]:
            d[edges[0][i]] += 1
            d[edges[1][i]] += 1

    nodes = []
    for i, year in enumerate(node_years):
        if year <= year_bound[1]:
            nodes.append([i, d[i]])

    nodes.sort(key = take_second, reverse = True)

    nodes = nodes[: int(proportion * len(nodes))]

    result_edges = []
    result_features = []
    result_labels = []

    for node in nodes:
        result_features.append(graph['node_feat'][node[0]])
    result_features = np.array(result_features)

    ids = {}
    for i, node in enumerate(nodes):
        ids[node[0]] = i

    for i in range(edges.shape[1]):
        if edges[0][i] in ids and edges[1][i] in ids:
            result_edges.append([ids[edges[0][i]], ids[edges[1][i]]])
    result_edges = np.array(result_edges).transpose(1, 0)

    result_labels = dataset.labels[[node[0] for node in nodes]]

    edge_index = torch.tensor(result_edges, dtype=torch.long)
    node_feat = torch.tensor(result_features, dtype=torch.float)

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': node_feat.size(0)}

    dataset.label = torch.tensor(result_labels)
    node_years_new = [node_years[node[0]] for node in nodes]
    dataset.test_mask = (torch.tensor(node_years_new) > year_bound[0])

    return dataset


def iid_split_index(num_nodes, seed):
    z = np.random.RandomState(seed).uniform(0,1,num_nodes)
    rand_ = torch.from_numpy(z).float()
    train_mask = rand_<0.5
    val_mask = rand_>=0.7
    test_mask = torch.logical_not(torch.logical_or(train_mask, val_mask))

    # split_idx = {'train': torch.nonzero(train_mask), 
    #              'valid': torch.nonzero(val_mask), 
    #              'test':  torch.nonzero(test_mask)}

    split_idx = torch.empty(num_nodes)
    split_idx[torch.nonzero(train_mask)] = 0
    split_idx[torch.nonzero(val_mask)] = 1
    split_idx[torch.nonzero(test_mask)] = 2
    return split_idx

def ood_split_index(node_feature, edge_index, num_nodes, seed):
    z = node_feature
    pca = PCA()
    z = pca.fit_transform(z)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(z)
    # print(kmeans.labels_.sum())
    if kmeans.labels_.sum() < (1-kmeans.labels_).sum():
        minor_mask = kmeans.labels_
    else:
        minor_mask = (1-kmeans.labels_).astype(int)

    split_idx = {'train': None, 'valid': None, 'test': None}
    split_idx['test'] =  torch.nonzero(torch.from_numpy(minor_mask)).flatten()
    test_mask = torch.from_numpy(minor_mask)

    train_mask = torch.rand(num_nodes)>0.4
    val_mask = torch.logical_not(train_mask)
    train_mask = (torch.logical_and(torch.logical_not(test_mask.bool()), train_mask)).long()
    val_mask = (torch.logical_and(torch.logical_not(test_mask.bool()), val_mask)).long()
    # split_idx['train'] = torch.nonzero((torch.logical_and(torch.logical_not(torch.from_numpy(minor_mask).bool()), train_mask)).long()).flatten()
    # split_idx['valid'] = torch.nonzero((torch.logical_and(torch.logical_not(torch.from_numpy(minor_mask).bool()), val_mask)).long()).flatten()

    split_idx = torch.empty(num_nodes)
    split_idx[torch.nonzero(train_mask)] = 0
    split_idx[torch.nonzero(val_mask)] = 1
    split_idx[torch.nonzero(test_mask)] = 2

    return split_idx

def filter_samples(node_idxes, task:str=None, max_num_edges=-1, max_num_nodes=-1, root=None):
    '''
        avoid OOM some node has too many edges and neighbors
    '''
    # print(f"Filtering: max_num_edges:{max_num_edges}; max_num_nodes:{max_num_nodes}.")

    df = pd.read_csv(f'{root}/{task}_num_edges_nodes.txt',names=['id','num_nodes','num_edges'])

    # target nodes should be filtered
    if max_num_edges != -1: # too much edges
        df = df.loc[df['num_edges']<max_num_edges]
    if max_num_nodes != -1: # too much neighbors
        df = df.loc[df['num_nodes']<max_num_nodes]
    retain_samples = set(df['id'])

    # print(f"Original size: {node_idxes.shape}", end='; ')
    node_idxes = torch.tensor(list(
        filter(lambda x: x in retain_samples, node_idxes.numpy().tolist())))
    # print(f"Filtered size: {node_idxes.shape}")
    return node_idxes


def load_dataset(task, root='dataset/', max_num_edges=-1, max_num_nodes=-1, ood=False, seed=None):
    '''
    '''
    if task == 'Arxiv':
        dataset = PygNodePropPredDataset(root=root, name = 'ogbn-arxiv') 
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = dataset[0] # pyg graph object
        graph.edge_index = to_undirected(graph.edge_index)
        del graph.node_year
        num_classes = 40

        split_idx = {
            'train': train_idx, 
            'valid': valid_idx, 
            'test' : test_idx
        }
        print(f"{graph=}")
        print(f"{split_idx=}")
        return graph, split_idx, num_classes
    
    if task == 'ArxivOOD':

        def get_dataset(dataset, year=None):
            ### Load and preprocess data ###
            if dataset == 'ogb-arxiv':
                dataset = load_nc_dataset(data_dir=root, dataname='ogb-arxiv', year=year)
            else:
                raise ValueError('Invalid dataname')

            if len(dataset.label.shape) == 1:
                dataset.label = dataset.label.unsqueeze(1)

            dataset.n = dataset.graph['num_nodes']
            dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
            dataset.d = dataset.graph['node_feat'].shape[1]

            dataset.graph['edge_index'], dataset.graph['node_feat'] = \
                dataset.graph['edge_index'], dataset.graph['node_feat']

            return dataset
        
        tr_year, val_year, te_years = [[1950, 2011]], [[2011, 2014]], [[2014, 2016], [2016, 2018], [2018, 2020]]

        if os.path.exists(f"{root}/ArxivOODTrain.pt"):
            dataset_tr = torch.load(f"{root}/ArxivOODTrain.pt")
        else:
            dataset_tr = get_dataset(dataset='ogb-arxiv', year=tr_year[0])
            torch.save(dataset_tr, f"{root}/ArxivOODTrain.pt")

        if os.path.exists(f"{root}/ArxivOODValid.pt"):
            dataset_val = torch.load(f"{root}/ArxivOODValid.pt")
        else:
            dataset_val = get_dataset(dataset='ogb-arxiv', year=val_year[0])
            torch.save(dataset_val, f"{root}/ArxivOODValid.pt")
        
        if os.path.exists(f"{root}/ArxivOODTest.pt"):
            datasets_te = torch.load(f"{root}/ArxivOODTest.pt")
        else:
            datasets_te = [get_dataset(dataset='ogb-arxiv', year=te_years[i]) for i in range(len(te_years))]
            torch.save(datasets_te, f"{root}/ArxivOODTest.pt")

        # print(f"Train num nodes {dataset_tr.n} | target nodes {dataset_tr.test_mask.sum()} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
        # print(f"Val num nodes {dataset_val.n} | target nodes {dataset_val.test_mask.sum()} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
        # for i in range(len(te_years)):
        #     dataset_te = datasets_te[i]
        #     print(f"Test {i} num nodes {dataset_te.n} | target nodes {dataset_te.test_mask.sum()} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

        return dataset_tr, dataset_val, datasets_te


    if task == 'MAG':
        dataset = PygNodePropPredDataset(root=root, name = 'ogbn-mag') 
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = dataset[0]
        graph = Data(x=graph.x_dict['paper'], edge_index=to_undirected(graph.edge_index_dict[('paper', 'cites', 'paper')]),y=graph.y_dict['paper']) 

        num_classes = 349

        split_idx = {
            'train': train_idx, 
            'valid': valid_idx, 
            'test' : test_idx
        }

        return graph, split_idx, num_classes


    if task == 'Flickr':
        data = Flickr(root=f"{root}/Flickr")[0]
        # print(data)
        num_classes = 7
        del data.train_mask
        del data.val_mask
        del data.test_mask

    if task == 'Cora':
        data = CitationFull(root = f"{root}/Cora", name='Cora')[0]
        # print(data)
        # if ood:
        #     split_idx = ood_split_index(edge_index=data.edge_index, num_nodes=data.x.shape[0])
        # else:
        #     split_idx = iid_split_index(num_nodes=data.x.shape[0])
        num_classes = 70

    if task == 'Cora_ML':
        data = CitationFull(root = f"{root}/Cora_ML", name='Cora_ML')[0]
        # print(data)
        # if ood:
        #     split_idx = ood_split_index(edge_index=data.edge_index, num_nodes=data.x.shape[0])
        # else:
        #     split_idx = iid_split_index(num_nodes=data.x.shape[0])
        num_classes = 7

    if task == 'CiteSeer':
        data = CitationFull(root = f"{root}/CiteSeer", name='CiteSeer')[0]
        # print(data)
        # if ood:
        #     split_idx = ood_split_index(edge_index=data.edge_index, num_nodes=data.x.shape[0])
        # else:
        #     split_idx = iid_split_index(num_nodes=data.x.shape[0])
        num_classes = 6
    
    if task == 'PubMed':
        data = CitationFull(root = f"{root}/PubMed", name='PubMed')[0]
        # print(data)
        # if ood:
        #     split_idx = ood_split_index(edge_index=data.edge_index, num_nodes=data.x.shape[0])
        # else:
        #     split_idx = iid_split_index(num_nodes=data.x.shape[0])
        num_classes = 3
    
    if task == 'DBLP':
        data = CitationFull(root = f"{root}/DBLP", name='DBLP')[0]
        # print(data)
        # if ood:
        #     split_idx = ood_split_index(edge_index=data.edge_index, num_nodes=data.x.shape[0])
        # else:
        #     split_idx = iid_split_index(num_nodes=data.x.shape[0])
        num_classes = 4

    if task == 'Computers':
        data = Amazon(root = f"{root}/Amazon", name='Computers')[0]
        # print(data)
        # if ood:
        #     split_idx = ood_split_index(edge_index=data.edge_index, num_nodes=data.x.shape[0])
        # else:
        #     split_idx = iid_split_index(num_nodes=data.x.shape[0])
        num_classes = 10
    
    if task == 'Photo':
        data = Amazon(root = f"{root}/Amazon", name='Photo')[0]
        # print(data)
        # if ood:
        #     split_idx = ood_split_index(edge_index=data.edge_index, num_nodes=data.x.shape[0])
        # else:
        #     split_idx = iid_split_index(num_nodes=data.x.shape[0])
        num_classes = 8
    
    if task == 'SyntheticColor':
        data = torch.load(f'{root}/Synthetic.pt')
        data.y = data.y_color
        num_classes = 3

    if task == 'SyntheticShape':
        data = torch.load(f'{root}/Synthetic.pt')
        data.y = data.y_shape
        num_classes = 2

    
    if task == 'SyntheticColorV2':
        data = torch.load(f'{root}/SyntheticV2.pt')
        data.y = data.y_color
        num_classes = 2

    if task == 'SyntheticShapeV2':
        data = torch.load(f'{root}/SyntheticV2.pt')
        data.y = data.y_shape
        num_classes = 2

    ################################# GOOD Dataset Start #########################################

    if task == 'WebKBCovariate':
        dataset, dataset_info = GOODWebKB.load(dataset_root=root, domain="university", shift="covariate")
        data = dataset[0]
        
        # Data(x=[617, 1703], edge_index=[2, 1138], y=[617], university=[617], train_mask=[617], val_mask=[617], 
        # test_mask=[617], id_val_mask=[617], id_test_mask=[617], env_id=[617], domain='university', domain_id=[617])
        num_classes = 5
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.university
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    if task == 'WebKBConcept':
        dataset, dataset_info = GOODWebKB.load(dataset_root=root, domain="university", shift="concept")
        data = dataset[0]
        
        # Data(x=[617, 1703], edge_index=[2, 1138], y=[617], university=[617], train_mask=[617], val_mask=[617], 
        # test_mask=[617], id_val_mask=[617], id_test_mask=[617], env_id=[617], domain='university', domain_id=[617])
        num_classes = 5
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.university
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    if task == 'CBASCovariate':
        dataset, dataset_info = GOODCBAS.load(dataset_root=root, domain="color", shift="covariate")
        data = dataset[0]
        
        # Data(x=[700, 4], edge_index=[2, 3962], y=[700], expl_mask=[700], edge_label=[3962], train_mask=[700], 
        # val_mask=[700], test_mask=[700], id_val_mask=[700], id_test_mask=[700], env_id=[700], domain_id=[700])
        num_classes = 4
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.expl_mask
        del data.edge_label
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain_id

        return data, split_idx, num_classes
    
    if task == 'CBASConcept':
        dataset, dataset_info = GOODCBAS.load(dataset_root=root, domain="color", shift="concept")
        data = dataset[0]
        
        # Data(x=[700, 4], edge_index=[2, 3962], y=[700], expl_mask=[700], edge_label=[3962], train_mask=[700], 
        # val_mask=[700], test_mask=[700], id_val_mask=[700], id_test_mask=[700], env_id=[700], domain_id=[700])
        num_classes = 4
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.expl_mask
        del data.edge_label
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain_id

        return data, split_idx, num_classes

    if task == 'CoraWordCovariate':
        dataset, dataset_info = GOODCora.load(dataset_root=root, domain="word", shift="covariate")
        data = dataset[0]
        
        # Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793], word=[19793], train_mask=[19793], 
        # val_mask=[19793], test_mask=[19793], id_val_mask=[19793], id_test_mask=[19793], env_id=[19793], domain='word', domain_id=[19793])
        num_classes = 70
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.word
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    if task == 'CoraWordConcept':
        dataset, dataset_info = GOODCora.load(dataset_root=root, domain="word", shift="concept")
        data = dataset[0]
        
        # Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793], word=[19793], train_mask=[19793], 
        # val_mask=[19793], test_mask=[19793], id_val_mask=[19793], id_test_mask=[19793], env_id=[19793], domain='word', domain_id=[19793])
        num_classes = 70
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.word
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes


    if task == 'CoraDegreeCovariate':
        dataset, dataset_info = GOODCora.load(dataset_root=root, domain="degree", shift="covariate")
        data = dataset[0]
        
        # Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793], word=[19793], train_mask=[19793], 
        # val_mask=[19793], test_mask=[19793], id_val_mask=[19793], id_test_mask=[19793], env_id=[19793], domain='word', domain_id=[19793])
        num_classes = 70
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.degree
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    if task == 'CoraDegreeConcept':
        dataset, dataset_info = GOODCora.load(dataset_root=root, domain="degree", shift="concept")
        data = dataset[0]
        
        # Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793], word=[19793], train_mask=[19793], 
        # val_mask=[19793], test_mask=[19793], id_val_mask=[19793], id_test_mask=[19793], env_id=[19793], domain='word', domain_id=[19793])
        num_classes = 70
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.degree
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    
    if task == 'ArxivTimeCovariate':
        dataset, dataset_info = GOODArxiv.load(dataset_root=root, domain="time", shift="covariate")
        data = dataset[0]
        
        # Data(num_nodes=169343, edge_index=[2, 2315598], x=[169343, 128], node_year=[169343, 1], y=[169343], 
        # time=[169343], train_mask=[169343], val_mask=[169343], test_mask=[169343], id_val_mask=[169343], 
        # id_test_mask=[169343], env_id=[169343], domain='time', domain_id=[169343])
        
        num_classes = 40
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.node_year
        del data.time
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    if task == 'ArxivTimeConcept':
        dataset, dataset_info = GOODArxiv.load(dataset_root=root, domain="time", shift="concept")
        data = dataset[0]
        
        # Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793], word=[19793], train_mask=[19793], 
        # val_mask=[19793], test_mask=[19793], id_val_mask=[19793], id_test_mask=[19793], env_id=[19793], domain='word', domain_id=[19793])
        num_classes = 40
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.node_year
        del data.time
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes


    if task == 'ArxivDegreeCovariate':
        dataset, dataset_info = GOODArxiv.load(dataset_root=root, domain="degree", shift="covariate")
        data = dataset[0]
        
        # Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793], word=[19793], train_mask=[19793], 
        # val_mask=[19793], test_mask=[19793], id_val_mask=[19793], id_test_mask=[19793], env_id=[19793], domain='word', domain_id=[19793])
        num_classes = 40
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.node_year
        del data.degree
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    if task == 'ArxivDegreeConcept':
        dataset, dataset_info = GOODArxiv.load(dataset_root=root, domain="degree", shift="concept")
        data = dataset[0]
        
        # Data(x=[19793, 8710], edge_index=[2, 126842], y=[19793], word=[19793], train_mask=[19793], 
        # val_mask=[19793], test_mask=[19793], id_val_mask=[19793], id_test_mask=[19793], env_id=[19793], domain='word', domain_id=[19793])
        num_classes = 40
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.node_year
        del data.degree
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes
    
    
    if task == 'TwitchCovariate':
        dataset, dataset_info = GOODTwitch.load(dataset_root=root, domain="language", shift="covariate")
        data = dataset[0]
        
        # Data(x=[34120, 128], edge_index=[2, 892346], y=[34120, 1], language=[34120], train_mask=[34120],
        # val_mask=[34120], test_mask=[34120], id_val_mask=[34120], id_test_mask=[34120], env_id=[34120], 
        # domain='language', domain_id=[34120])
        
        num_classes = 2
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.language
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes

    if task == 'TwitchConcept':
        dataset, dataset_info = GOODTwitch.load(dataset_root=root, domain="language", shift="concept")
        data = dataset[0]
        
        # Data(x=[34120, 128], edge_index=[2, 892346], y=[34120, 1], language=[34120], train_mask=[34120],
        # val_mask=[34120], test_mask=[34120], id_val_mask=[34120], id_test_mask=[34120], env_id=[34120], 
        # domain='language', domain_id=[34120])
        
        num_classes = 2
        if ood:
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.val_mask).nonzero().flatten(), 
                'test' : (data.test_mask).nonzero().flatten()
            }
        else: # iid
            split_idx = {
                'train': (data.train_mask).nonzero().flatten(), 
                'valid': (data.id_val_mask).nonzero().flatten(), 
                'test' : (data.id_test_mask).nonzero().flatten()
            }

        del data.language
        del data.train_mask
        del data.val_mask
        del data.test_mask
        del data.id_val_mask
        del data.id_test_mask
        del data.env_id
        del data.domain
        del data.domain_id

        return data, split_idx, num_classes


    ################################# GOOD Dataset End #########################################


    if ood:
        if os.path.exists(f"{root}/splits/{task}-ood.pt"):
            split_idx = torch.load(f"{root}/splits/{task}-ood.pt")[:,seed]
        else:
            split_idxes = []
            for i in range(5):
                split_idx_i = ood_split_index(data.x, data.edge_index, data.x.size(0), seed=i).reshape(-1,1)
                split_idxes.append(split_idx_i)
            split_idxes = torch.cat(split_idxes,dim=1)
            torch.save(split_idxes, f"{root}/splits/{task}-ood.pt")
        split_idx = torch.load(f"{root}/splits/{task}-ood.pt")[:,seed]

    else:
        if os.path.exists(f"{root}/splits/{task}-iid.pt"):
            split_idx = torch.load(f"{root}/splits/{task}-iid.pt")[:,seed]
        else:
            split_idxes = []
            for i in range(5):
                split_idx_i = iid_split_index(data.x.size(0), seed=i).reshape(-1,1)
                split_idxes.append(split_idx_i)
            split_idxes = torch.cat(split_idxes,dim=1)
            torch.save(split_idxes, f"{root}/splits/{task}-iid.pt")
        
        split_idx = torch.load(f"{root}/splits/{task}-iid.pt")[:,seed]
        
    split_idx = {
        'train': (split_idx==0).nonzero().flatten(), 
        'valid': (split_idx==1).nonzero().flatten(), 
        'test' : (split_idx==2).nonzero().flatten()
    }

    return data, split_idx, num_classes


@torch.no_grad()
def pred_test(model, data, split_idx, evaluator):
    model.eval()
    with torch.no_grad():
        out = model(data)
        out = out.log_softmax(dim=-1)
        y_pred = out.argmax(dim=-1, keepdim=True)
        # print(y_pred.shape)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


if __name__ == "__main__":

    for dataset in ['Arxiv','ArxivOOD']: # , "Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed", "Computers", "Photo", "Flickr", "SyntheticShape", "SyntheticColor"
        print(dataset)
        for seed in range(5):
            for ood in [False, True]:
                load_dataset(dataset, root='../dataset/', ood=ood, seed=seed)
