import warnings
warnings.filterwarnings('ignore')

import json

import os
import re
from pathlib import Path
import argparse

import random
random.seed(0)

import time
import copy
from tqdm import tqdm

import math
import pandas as pd
import numpy as np
np.random.seed(0)

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from sklearn.metrics import classification_report

from cftm import HomoStructCFTM
from pipeline.pipeline import Pipeline

from utils.EgoGraphGenerator import PYGEgoSampledGraphDataset
from utils.EarlyStopper import EarlyStopper
from utils.JSD import js_loss
from utils.Dataset import load_dataset


def eval_(y_true, y_pred):
    metrics = classification_report(y_true=y_true, y_pred=y_pred, digits=6, output_dict=True, zero_division=0)
    metrics = {'acc': metrics['accuracy'], 'macro_f1': metrics['macro avg']['f1-score']}
    return metrics

def train_joint(model, cftm, dataloader, optimizer, device, accumulation_steps):
    '''
    '''
    model.train()
    cftm.train()

    for step, batch_data in enumerate(tqdm(dataloader, desc='Train', leave=False)):
        # Compute how many batches are left in the epoch
        batches_left = len(dataloader) - step
        # Skip incomplete accumulation steps at the end of the epoch
        if batches_left < accumulation_steps:
            break

        batch_data = batch_data.to(device)
        # optimizer.zero_grad()

        # training=True to get continuous value(range 0 to 1)
        # training True [num_edge, num_head]; training False [num_edge]
        # print(batch_data)
        crucial_mask = cftm(batch_data, training=True)

        crucial_mask = crucial_mask.mean(dim=-1)
        nonsense_mask = 1. - crucial_mask

        # print(tmp_crucial_mask.shape)
        ori_out = model(batch_data, edge_weight=None)
        pos_out = model(batch_data, edge_weight=crucial_mask)
        neg_out = model(batch_data, edge_weight=nonsense_mask)

        # logits of ori, pos, neg inputs -> [batch_size, num_classes]
        targets_ori_out = ori_out[batch_data['center_mask'].flatten().bool()]
        targets_pos_out = pos_out[batch_data['center_mask'].flatten().bool()]
        targets_neg_out = neg_out[batch_data['center_mask'].flatten().bool()]
        # log probs of ori, pos, neg inputs -> [batch_size, num_classes]
        targets_ori_out_prob = targets_ori_out.log_softmax(dim=-1)
        targets_pos_out_prob = targets_pos_out.log_softmax(dim=-1)
        targets_neg_out_prob = targets_neg_out.log_softmax(dim=-1)
        # y_true
        y = batch_data['y'][batch_data.center_mask.flatten().bool()].flatten()

        # prediction loss
        pred_loss = F.nll_loss(targets_ori_out_prob, y)

        # CFTM loss
        cftm_loss = cftm._loss(
            crucial_masked_pred =  targets_pos_out_prob,
            nonsense_masked_pred = targets_neg_out_prob, 
            y_true = y, 
            crucial_mask = crucial_mask,
        )
        

        # CL Loss
        pos_sim = 1-js_loss(targets_ori_out, targets_pos_out)
        neg_sim = 1-js_loss(targets_ori_out, targets_neg_out)
        pos_sim, neg_sim = torch.exp(pos_sim/args['cont_temp']), torch.exp(neg_sim/args['cont_temp'])

        cont_loss = (-torch.log(pos_sim/(pos_sim+neg_sim))).mean()

        # total loss
        # print(f"{step=}, {pred_loss=}, {cont_loss=}, {cftm_loss=}")
        loss = pred_loss + cont_loss + cftm_loss 
        loss = loss / accumulation_steps # Normalize loss (if averaged)
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            try:
                # clip gradient
                gnn_gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                cftm_gradient_norm = nn.utils.clip_grad_norm_(cftm.parameters(), max_norm=2.0, error_if_nonfinite=True)
                optimizer.step()  # update
                optimizer.zero_grad() 
            except:
                optimizer.zero_grad() 
        # optimizer.step()


@torch.no_grad()
def cftm_test(model, cftm, dataloader, device):
    '''
    '''
    def remove_edge(edge_index, mask):
        '''
        '''
        return edge_index[:,mask]
    
    def eval_(y_true, y_pred):
        metrics = classification_report(y_true=y_true, y_pred=y_pred, digits=6, output_dict=True, zero_division=0)
        metrics = {'acc': metrics['accuracy'], 'macro_f1': metrics['macro avg']['f1-score']}
        return metrics

    def iter_dataloader(_dataloader):
        '''
        '''
        results = {
            'ori' : [],
            'pos' : [],
            'neg' : [],
            'true': [],
        }

        mask_rate = {
            '1': [],
            '0': []
        }
        with torch.no_grad():
            for batch_data in tqdm(_dataloader, desc='Infer', leave=False):
                batch_data = batch_data.to(device)

                # training=False to make mask binarized, training=True to make mask continuous,
                crucial_mask = cftm(batch_data, training=False)
                # print(crucial_mask, nonsense_mask)

                crucial_mask = crucial_mask.mean(dim=-1)
                # nonsense_mask = nonsense_mask.mean(dim=-1)

                # print(crucial_mask, nonsense_mask)

                crucial_mask = crucial_mask>0.5
                # nonsense_mask = nonsense_mask>=0.4
                nonsense_mask = torch.logical_not(crucial_mask)

                # computing rate
                crucial_rate = crucial_mask.sum()/crucial_mask.shape[0]
                nonsense_rate = nonsense_mask.sum()/nonsense_mask.shape[0]
                mask_rate['1'].append(crucial_rate.item())
                mask_rate['0'].append(nonsense_rate.item())

                # logits of ori, pos, neg inputs -> [num_nodes, num_classes]
                ori_edge_index = batch_data.edge_index
                ori_out = model(batch_data)

                pos_edge_index = remove_edge(ori_edge_index, crucial_mask)
                batch_data.edge_index = pos_edge_index
                pos_out = model(batch_data)
                
                neg_edge_index = remove_edge(ori_edge_index, nonsense_mask)
                batch_data.edge_index = neg_edge_index
                neg_out = model(batch_data)

                # logits of ori, pos, neg inputs -> [batch_size, num_classes]
                targets_ori_out = ori_out[batch_data['center_mask'].flatten().bool()]
                targets_pos_out = pos_out[batch_data['center_mask'].flatten().bool()]
                targets_neg_out = neg_out[batch_data['center_mask'].flatten().bool()]

                # predicted class -> [batch_size, 1]
                targets_ori_y = targets_ori_out.argmax(dim=-1, keepdim=False)
                targets_pos_y = targets_pos_out.argmax(dim=-1, keepdim=False)
                targets_neg_y = targets_neg_out.argmax(dim=-1, keepdim=False)

                # y_true
                y = batch_data['y'][batch_data.center_mask.flatten().bool()].flatten()
                
                results['ori'].append(targets_ori_y)
                results['pos'].append(targets_pos_y)
                results['neg'].append(targets_neg_y)
                results['true'].append(y)
        
        # concat batches
        results = {k:torch.cat(v,dim=0) for k,v in results.items()} # .reshape(-1,1)
        # mean of crucial/non-crucial rate
        mask_rate = {k: np.mean(v) for k,v in mask_rate.items()}
        return results, mask_rate
    
    model.eval()
    cftm.eval()
    _results, _mask_rate = iter_dataloader(_dataloader = dataloader)

    ori_metric = eval_(
        y_true= _results['true'].cpu().numpy(), 
        y_pred= _results['ori'].cpu().numpy()
    )
    
    pos_metric = eval_(
        y_true= _results['true'].cpu().numpy(), 
        y_pred= _results['pos'].cpu().numpy()
    )

    neg_metric = eval_(
        y_true= _results['true'].cpu().numpy(), 
        y_pred= _results['neg'].cpu().numpy()
    )

    cftm_metric = (pos_metric['acc']/ori_metric['acc'] + 1)/(_mask_rate['1']+1) \
             - (neg_metric['acc']/ori_metric['acc'] + 1)/(_mask_rate['0']+1)
            #  + (_mask_rate['1']+1)/(_mask_rate['0']+1)

    return ori_metric, pos_metric, neg_metric, _mask_rate, cftm_metric

def run(args):

    # get pyg data
    device = torch.device(f"cuda:{args['device']}" if args['device']>=0 else "cpu")

    task = args['task']

    s = time.time()
    data, split_idx, num_classes = load_dataset(
        task=task, root='dataset/', 
        max_num_edges= args['max_num_edges'], max_num_nodes=args['max_num_nodes'],
        ood=args['ood'], seed = args['round']
    )
    print(f"Dataset: {task}; Loading time: {time.time()-s:.2f}s; ", end='')
    print(f"Split: {'; '.join([f'{k}-{v.shape[0]}' for k,v in split_idx.items()])}")
    
    # data = data.to(device)
    
    num_train_samples = split_idx['train'].shape[0]
    num_used_train_samples = int(split_idx['train'].shape[0]*args['data_ratio'])
    split_idx['train'] = split_idx['train'][
         torch.randperm(
             num_train_samples
         )[:num_used_train_samples]
    ]

    train_dataset = PYGEgoSampledGraphDataset(
        pyg=data, num_hops=args['num_hops'], idxes=split_idx['train'], 
        num_neighbors=[args['num_neighbors']]*args['num_hops'],
        rw_feat=args['rw_feat'], cache=True
    )

    valid_dataset = PYGEgoSampledGraphDataset(
        pyg=data, num_hops=args['num_hops'], idxes=split_idx['valid'],
        num_neighbors=[args['num_neighbors']]*args['num_hops'],
        rw_feat=args['rw_feat'], cache=True
    )

    test_dataset  = PYGEgoSampledGraphDataset(
        pyg=data, num_hops=args['num_hops'], idxes=split_idx['test'],
        num_neighbors=[args['num_neighbors']]*args['num_hops'],
        rw_feat=args['rw_feat'], cache=True
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True , drop_last = True , batch_size=args['batch_size'], num_workers=0, collate_fn=Batch.from_data_list)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, drop_last = False, batch_size=args['batch_size'], num_workers=0, collate_fn=Batch.from_data_list)
    test_dataloader  = DataLoader(test_dataset , shuffle=False, drop_last = False, batch_size=args['batch_size'], num_workers=0, collate_fn=Batch.from_data_list)

    # GNN encoder + classifier pipeline
    pipeline = Pipeline(
        encoder = args['encoder'], 
        in_dim  = data.x.shape[1], 
        hid_dim = args['hid_dim'],
        out_dim = num_classes,
        num_hops = args['num_hops']
    )

    print(pipeline)

    n_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    pipeline = pipeline.to(device)
    in_dim = data.x.shape[1]

    cftm = HomoStructCFTM(
        node_x_dim = in_dim, 
        node_rw_dim = args['num_hops'],
        node_ef_dim = 2,
        hid_dim=args['hid_dim'], 
        reg_coefs=[args['size_reg'], 0.], sample_bias=0.,
        rw_feat=args['rw_feat'], ego_flag=args['ego_flag'], args=args
    ).to(device)
    print(cftm)
    
    optimizer = torch.optim.AdamW(list(cftm.parameters())+list(pipeline.parameters()), lr=args['lr'])
    # cftm_opt = torch.optim.AdamW(cftm.parameters(), lr=args['lr'])
    
    # early stop
    es = EarlyStopper(min_step=args['min_step'], max_step=args['max_step'], patience=args['patience'], improve='up')
    run_times = []
    for epoch in range(1000):
        train_time_s = time.time()
        train_joint(model = pipeline, cftm = cftm, dataloader = train_dataloader, optimizer = optimizer, device = device, accumulation_steps=args['accumulation_steps'])
        train_time_e = time.time()
        epoch_train_time = train_time_e-train_time_s
        run_times.append(epoch_train_time)
        # to change sampling temperature
        # cftm.power += 1/20
        torch.cuda.empty_cache()

        valid_time_s = time.time()
        result = cftm_test(pipeline, cftm, valid_dataloader, device)
        valid_time_e = time.time()
        epoch_valid_time = valid_time_e-valid_time_s
        ori_metric, pos_metric, neg_metric, _mask_rate, cftm_metric = result

        acc = ori_metric['acc']
        f1 = ori_metric['macro_f1']
        scores = {
            'acc':acc
        }
        
        print(
            f'Epoch: {epoch:02d}, Train/Valid Time: {epoch_train_time:.2f}/{epoch_valid_time:.2f}, '
            f'ori_acc: {ori_metric["acc"]:8.4%}, '
            f'pos_acc: {pos_metric["acc"]:8.4%}, '
            f'neg_acc: {neg_metric["acc"]:8.4%}, '
            # f'ori_f1: {ori_metric["macro_f1"]:8.4%}, '
            # f'pos_f1: {pos_metric["macro_f1"]:8.4%}, '
            # f'neg_f1: {neg_metric["macro_f1"]:8.4%}, '
            f'1_rate: {_mask_rate["1"]:8.4%}, '
            f'0_rate: {_mask_rate["0"]:8.4%}, '
            # f'metric: {cftm_metric:.4f}'
        )
        es.update(step=epoch, updated_score=scores, updated_model=(copy.deepcopy(pipeline), copy.deepcopy(cftm)))
        # es.update(step=epoch, updated_score=scores, updated_model=(pipeline, cftm))
        
        if es.stopped:
            break

    run_time = np.mean(run_times)

    # saving 
    for metric, (pipeline, cftm) in es.best_models.items():
        result = cftm_test(pipeline, cftm, test_dataloader, device)
        ori_metric, pos_metric, neg_metric, _mask_rate, cftm_metric = result
        acc = ori_metric['acc']
        f1 = ori_metric['macro_f1']
        
        print(f'#params {n_params:10d}; Runtime {run_time:6.2f}; Test Acc: {acc:.4%}; Test F1: {f1:.4%}; ')
        
        torch.save(
            {
                'pipeline': pipeline.state_dict(),
                'cftm': cftm.state_dict()
            },
            f"{args['saving_path']}/"
            f"round-{args['round']}_nparmas-{n_params}_runtime-{run_time:.2f}_"
            f"Acc-{acc:.4%}_F1-{f1:.4%}_"
            f"P-{_mask_rate['1']:.4%}_N-{_mask_rate['0']:.4%}_"
            f"Pacc-{pos_metric['acc']:.4%}_Pf1-{pos_metric['macro_f1']:.4%}_"
            f"Nacc-{neg_metric['acc']:.4%}_Nf1-{neg_metric['macro_f1']:.4%}"
            f".pt"
        )

        return n_params, run_time, acc, f1


def main(args):
    '''
    '''
    # GNN path
    dist_type = 'ood' if args['ood'] else 'iid'
    # gnn_path = f"./ckpts/backbone/{args['encoder']}/{args['task']}/{dist_type}/{args['data_ratio']}"
    # CFTM path
    saving_path = f"./ckpts/cftm_joint_save_both/rw_{args['rw_feat']}/ef_{args['ego_flag']}/{args['encoder']}/{args['task']}/{dist_type}/dr_{args['data_ratio']}/"\
                  f"size_reg_{args['size_reg']}/rank_coef_{args['rank_coef']}/cont_temp_{args['cont_temp']}"
    Path(saving_path).mkdir(parents=True, exist_ok=True)
    trained_models = os.listdir(saving_path)
    trained_models = set(map(lambda x: int(re.search('round-(?P<round>\d+)', x).group('round')), trained_models))
    args['saving_path'] = saving_path
    # args['gnn_path'] = gnn_path
    not_trained = {0, 1, 2, 3, 4} - trained_models # 
    if not_trained:
        for i in not_trained:
            args['round'] = i
            print(args)
            n_params, run_time, acc, f1 = run(args)
    else:
        print(saving_path)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-Param Setting')

    # parser.add_argument('--encoder', type=str, choices=['GraphSAGE', 'GAT', 'GIN'], default='GAT')
    # parser.add_argument('--task', type=str, choices=["SyntheticShape", "SyntheticColor","Arxiv","MAG","Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed","Computers","Photo","Flickr"], default='Cora_ML')
    parser.add_argument('--encoder', type=str, choices=['GraphSAGE', 'GAT', 'GIN', 'GCNGOOD'], default='GAT')
    
    parser.add_argument('--task', type=str, choices=[
                'TwitchCovariate','TwitchConcept','ArxivTimeCovariate','ArxivTimeConcept',
                'ArxivDegreeCovariate','ArxivDegreeConcept','CoraDegreeCovariate','CoraDegreeConcept',
                'CoraWordCovariate','CoraWordConcept','WebKBConcept','WebKBCovariate',
                'CBASConcept', 'CBASCovariate',
                "SyntheticShapeV2", "SyntheticColorV2", "Arxiv","MAG","Cora", "Cora_ML", 
                "CiteSeer", "DBLP", "PubMed", "Computers","Photo","Flickr"], default='Cora_ML')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--num_neighbors', type=int, default=32)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=96)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_ratio', type=float, default=1.)

    parser.add_argument('--ood', action='store_true')
    parser.add_argument('--device', type=int, default=-1)

    parser.add_argument('--min_step', type=int, default=10)
    parser.add_argument('--max_step', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)

    # CF Plugin params
    parser.add_argument('--rw_feat', action='store_true')
    parser.add_argument('--ego_flag', action='store_true')
    parser.add_argument('--size_reg', type=float, default=1.) # [2.0 ~ 4.0]
    parser.add_argument('--rank_coef', type=float, default=1.) # [0.4, 0.6, 0.8 ,1.0]
    parser.add_argument('--cont_temp', type=float, default=0.1) 

    # -1 for no filtering
    parser.add_argument('--max_num_edges', type=int, default=-1)
    parser.add_argument('--max_num_nodes', type=int, default=-1)
    
    args = parser.parse_args()
    args = vars(args)

    main(args)
    
    # lp = LineProfiler()
    # lp.add_function(train_joint)
    # lp.run('train_joint()')
    # lp.print_stats()