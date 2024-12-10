import torch
import torch_geometric as ptgeom
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import math


class ParallelMLP(nn.Module):
    def __init__(self, num_mlps, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(ParallelMLP, self).__init__()
        self.input_size = input_size
        self.num_mlps = num_mlps
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1

        # Create weights, biases, batch normalization, and dropout for each layer of each MLP
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            self.weights.append(nn.Parameter(torch.Tensor(num_mlps, layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(nn.Parameter(torch.Tensor(num_mlps, layer_sizes[i + 1])))
            if i < self.num_layers - 1: # output layer is not normalized or regularized
                self.batch_norms.append(nn.BatchNorm1d(num_mlps * layer_sizes[i + 1]))
                self.dropouts.append(nn.Dropout(p=self.dropout_prob))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with adjusted Xavier normal initialization
        for weight in self.weights:
            fan_in = weight.size(1)  # Get the second dimension size (ignoring num_mlps dimension)
            fan_out = weight.size(2)  # Get the third dimension size (ignoring num_mlps dimension)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            nn.init.uniform_(weight, -a, a)
        
        # Initialize biases to zero
        for bias in self.biases:
            nn.init.zeros_(bias)
  

    def forward(self, x):
        # x should be of shape (batch_size, input_size)
        # Pass input through each layer
        for i in range(self.num_layers):
            if i == 0:
                x = torch.einsum('bi,kih->bkh', x, self.weights[0])
            else:
                x = torch.einsum('bki,kih->bkh', x, self.weights[i])

            x = x + self.biases[i]

            if i < self.num_layers - 1: # not last layer 
                B, K, H = x.size()

                # Apply batch normalization, need to flatten and unflatten MLP dimensions
                x = x.view(B, K*H)
                x = self.batch_norms[i](x)
                x = x.view(B, K, H)

                # Apply non-linearity
                x = F.relu(x)

                # Apply dropout
                x = self.dropouts[i](x)

        return x


class ExplainerModel(nn.Module):
    def __init__(self, in_dim, hid_dim, num_head):
        '''
        '''
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_head = num_head

        self.m_row = nn.Linear(self.in_dim, self.hid_dim)
        self.m_col = nn.Linear(self.in_dim, self.hid_dim)
        self.m_ego = nn.Linear(self.in_dim, self.hid_dim)

        self.edge_emb_dropout = nn.Dropout(0.5)

        self.prob_estimator = ParallelMLP(
            num_mlps = self.num_head, 
            input_size = self.hid_dim, 
            hidden_sizes = [self.hid_dim], 
            output_size = 2, 
            dropout_prob = 0.5
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier (Glorot) normal initialization for weights
        nn.init.xavier_uniform_(self.m_row.weight)
        nn.init.xavier_uniform_(self.m_col.weight)
        nn.init.xavier_uniform_(self.m_ego.weight)
        # Set biases to zero
        if self.m_row.bias is not None: nn.init.zeros_(self.m_row.bias)
        if self.m_col.bias is not None: nn.init.zeros_(self.m_col.bias)
        if self.m_ego.bias is not None: nn.init.zeros_(self.m_ego.bias)
        
  
    def forward(self, row_embs, col_embs, center_embs):
        '''
        '''
        row_embs = self.m_row(row_embs)
        col_embs = self.m_col(col_embs)
        ego_embs = self.m_ego(center_embs)

        edge_embs = ego_embs + row_embs + col_embs
        

        input_expl = self.edge_emb_dropout(edge_embs)

        input_expl = self.prob_estimator(input_expl) # [num_edge, num_head, 2]

        return input_expl



class HomoStructCFTM(nn.Module):
    '''
    '''
    def __init__(
            self, 
            node_x_dim = None, 
            node_rw_dim = None,
            node_ef_dim = 2,
            hid_dim = None, 
            # temp=(5.0, 2.0), 
            reg_coefs=(2., 1.0), 
            cf_coef=2., rank_margin=1., power=0.0, sample_bias=0,
            rw_feat = False, ego_flag = False, args=None
        ):
        '''
        '''
        super().__init__()

        self.hid_dim = hid_dim
        self.rw_feat = rw_feat
        self.ego_flag = ego_flag

        self.node_x_dim = node_x_dim
        self.node_rw_dim = node_rw_dim
        self.node_ef_dim = node_ef_dim

        self.node_x_feat_transform = nn.Linear(self.node_x_dim, self.hid_dim)

        if self.rw_feat:
            self.node_rw_feat_transform = nn.Linear(self.node_rw_dim, self.hid_dim)

        if self.ego_flag:
            self.node_ef_feat_transform = nn.Linear(self.node_ef_dim, self.hid_dim)

        # self.node_emb_norm = nn.BatchNorm1d(self.hid_dim)
        self.node_emb_norm = nn.LayerNorm([self.hid_dim])

        self.explainer_model = ExplainerModel(in_dim=self.hid_dim, hid_dim=self.hid_dim, num_head=4)

        self.sample_bias = sample_bias
        self.reg_coefs = reg_coefs
        self.cf_coef = cf_coef
        self.rank_margin = rank_margin
        self.power = power

        self.args = args

        self.reset_parameters()
    
    def reset_parameters(self):
        # Xavier (Glorot) normal initialization for weights
        nn.init.xavier_uniform_(self.node_x_feat_transform.weight)
        # Set biases to zero
        if self.node_x_feat_transform.bias is not None:
            nn.init.zeros_(self.node_x_feat_transform.bias)

        if self.rw_feat:
            # Xavier (Glorot) normal initialization for weights
            nn.init.xavier_uniform_(self.node_rw_feat_transform.weight)
            # Set biases to zero
            if self.node_rw_feat_transform.bias is not None:
                nn.init.zeros_(self.node_rw_feat_transform.bias)

        if self.ego_flag:
            # Xavier (Glorot) normal initialization for weights
            nn.init.xavier_uniform_(self.node_ef_feat_transform.weight)
            # Set biases to zero
            if self.node_ef_feat_transform.bias is not None:
                nn.init.zeros_(self.node_ef_feat_transform.bias)

    def _create_explainer_input(self, data):
        '''
        '''
        node_x = data.x
        
        edge_index = data.edge_index
        node_batch_indicator = data.batch
        center_mask = data.center_mask

        node_embs = self.node_x_feat_transform(node_x)
       
        if self.rw_feat:
            rw_feat = data.rw_feat
            rw_feat_embs = self.node_rw_feat_transform(rw_feat)
            node_embs += rw_feat_embs
            
        if self.ego_flag:
            ef_feat = center_mask.long()
            ef_feat = F.one_hot(ef_feat, num_classes=2).float()
            ef_feat_embs = self.node_ef_feat_transform(ef_feat)
            node_embs += ef_feat_embs

        node_embs = self.node_emb_norm(node_embs)

        # get src and dst nodes embeddings of all edges
        rows = edge_index[0]
        cols = edge_index[1]
        row_embs = node_embs[rows]
        col_embs = node_embs[cols]

        # get ego node embeddings
        edge_batch = node_batch_indicator[rows] # identify which edge belong to which graph
        center_embs = node_embs[center_mask.flatten().bool()][edge_batch]

        return row_embs, col_embs, center_embs


    def _sample_graph(self, sampling_weights, tau=1., hard=False, eps=1e-10, dim=-1, training = None):
        
        assert training == self.training, 'training != self.training'
        if self.training:
            # -> [num_edge, num_head, 2]
            graph = F.gumbel_softmax(sampling_weights, tau=tau, hard=hard, eps=eps, dim=dim) 
            graph = graph[:,:,1] # -> [num_edge, num_head]
        else:
            # graph = graph.argmax(dim=-1).to(dtype=graph.dtype)
            graph = sampling_weights.softmax(dim=-1)[:,:,1]

        return graph

    def forward(self, data, training=True):
        '''
        '''
        row_embs, col_embs, center_embs = self._create_explainer_input(data)
        sampling_weights = self.explainer_model(row_embs, col_embs, center_embs)
        crucial_mask = self._sample_graph(sampling_weights, training=training) # bias=self.sample_bias, 

        return crucial_mask


    def _loss(self, crucial_masked_pred, nonsense_masked_pred, y_true, crucial_mask):
        """
        crucial_masked_pred, nonsense_masked_pred was after logsoftmax
        Returns the loss score based on the given mask.
        :return: loss
        """
        mask = crucial_mask

        # Size Loss (0~1)
        size_reg = self.reg_coefs[0]
        size_loss = torch.mean(mask) * size_reg

        # # Mask Entropy Loss (0~log2)
        # entropy_reg =  self.reg_coefs[1]
        # EPS = 0.005
        # mask = mask*0.99 + EPS # avoid zero log
        # mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Positive Prediction Loss
        cce_loss = F.nll_loss(crucial_masked_pred, y_true)
        # Negative Prediction loss 
        neg_cce_loss = F.nll_loss(nonsense_masked_pred, y_true)
        # Rank Loss
        rank_loss = self.args['rank_coef']*(cce_loss/neg_cce_loss)

        return size_loss + cce_loss + neg_cce_loss + rank_loss # + mask_ent_loss