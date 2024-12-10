from importlib import import_module

import torch
import torch.nn as nn


class classifier(nn.Module):
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int):
        '''
        '''
        super().__init__()

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_dim, hid_dim),

            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, out_dim),
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    def forward(self, x):
        x = self.mlp(x)
        return x


class Pipeline(nn.Module):
    def __init__(self, encoder:str, in_dim:int, hid_dim:int, out_dim:int, num_hops:int, num_heads:int=None):
        '''
        '''
        super().__init__()
        self.out_dim = out_dim
        Config = import_module(f'gnns.{encoder}').Config
        self.encoder_config = Config()
        self.encoder_config.in_dim = in_dim
        self.encoder_config.hid_dim = hid_dim
        self.encoder_config.num_layers = num_hops
        self.encoder_config.heads = num_heads

        model = import_module(f'gnns.{encoder}').Net
        self.encoder = model(config=self.encoder_config)

        self.clf = classifier(
             in_dim = self.encoder_config.hid_dim, 
             hid_dim = self.encoder_config.hid_dim,
             out_dim = self.out_dim, 
        )


    def forward(self, data, edge_weight=None):
        '''
        '''
        x = self.encoder(data, edge_weight)
        x = self.clf(x)
        return x
