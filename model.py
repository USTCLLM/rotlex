
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearUnit(nn.Module):
    def __init__(self, input_dim, out_dim, active_op='prelu', use_batch_norm=False, drop_rate = 1e9):
        super(LinearUnit, self).__init__()
        self.fc = nn.Linear(input_dim, out_dim)
        init_mean = 0.0
        init_stddev = 1.
        init_value = (init_stddev * np.random.randn(out_dim, input_dim).astype(
            np.float32) + init_mean) / np.sqrt(input_dim)
        self.fc.weight.data = torch.from_numpy(init_value)
        self.fc.bias.data = torch.zeros(out_dim)+0.1
        if active_op == "relu":
            self.activate = torch.nn.ReLU()
        elif active_op == "prelu":
            self.activate = torch.nn.PReLU()
        elif active_op == "lrelu":
            self.activate = torch.nn.LeakyReLU()
        elif active_op == "swish":
            self.activate = torch.nn.SiLU()
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
            bn_gamma=1.0
            bn_beta=0.0
            self.bn.weight.data = torch.ones(out_dim)*bn_gamma
            self.bn.bias.data = torch.ones(out_dim)*bn_beta
        if drop_rate<1:
            self.drop = nn.Dropout()
    
    def forward(self, input_data):
        hidden=self.fc(input_data)
        if hasattr(self, "bn"):
            hidden=self.bn(hidden)
        if hasattr(self, "drop"):
            hidden = self.drop(hidden)
        return self.activate(hidden)
    
class QueryEncoder(nn.Module):
    def __init__(self, in_dim, layer_dims:list, code_emb_dim, active_op:str, use_batch_norm, drop_rate):
        super(QueryEncoder, self).__init__()
        self.linearlist = nn.Sequential()
        for i in range(len(layer_dims)):
            self.linearlist.append(LinearUnit(in_dim, layer_dims[i], active_op, use_batch_norm, drop_rate))
            in_dim = layer_dims[i]
        self.linearlist.append(LinearUnit(in_dim, code_emb_dim, active_op, use_batch_norm, drop_rate))
        
    def forward(self, x):
        return self.linearlist(x)

class DeepModel(nn.Module):
    def __init__(self, in_dim, layer_dims:list, num_code, num_classes, code_emb_dim, active_op:str, use_batch_norm = True, drop_rate = 1e9) -> None:
        super(DeepModel, self).__init__()
        self.num_classes = num_classes
        self.enc = QueryEncoder(in_dim, layer_dims, code_emb_dim, active_op, use_batch_norm, drop_rate)
        self.emb = nn.Embedding(num_code, code_emb_dim)
        torch.nn.init.kaiming_uniform_(self.emb.weight, a=math.sqrt(5))

    # duck typing, so that we can use the same code for both DeepModel and DeepModelSepEnc
    def encode(self, batch_user_embedding,  i = 0):
        return self.enc(batch_user_embedding)
    
    def forward(self, batch_user_embedding, batch_item_idx, i = 0):
        batch_item_embedding = self.emb(batch_item_idx)
        return (self.enc(batch_user_embedding) * batch_item_embedding).sum(dim = 1)
    
    def forward_prob(self, batch_user_embedding, batch_item_idx, i = 0):
        return self.enc(batch_user_embedding) @ self.emb(batch_item_idx).T
    
    def classify_rg(self, batch_user_embedding, st, ed, i = 0):
        return self.enc(batch_user_embedding) @ self.emb.weight[st:ed].T
    
    def classify_all(self, batch_user_embedding, i = 0):
        return self.enc(batch_user_embedding) @ self.emb.weight.T
    
class DeepModelSepEnc(nn.Module):
    def __init__(self, in_dim, layer_dims:list, num_code, num_classes, code_emb_dim, active_op:str, use_batch_norm = True, drop_rate = 1e9) -> None:
        super(DeepModelSepEnc, self).__init__()
        self.num_classes = num_classes
        self.enc_sep = nn.ModuleList()
        for _ in range(2):
            self.enc_sep.append(QueryEncoder(in_dim, layer_dims, code_emb_dim, active_op, use_batch_norm, drop_rate))
        self.emb = nn.Embedding(num_code, code_emb_dim)
        torch.nn.init.kaiming_uniform_(self.emb.weight, a=math.sqrt(5))
        
    def encode(self, batch_user_embedding, i = 0):
        return self.enc_sep[i](batch_user_embedding)
    
    def forward(self, batch_user_embedding, batch_item_idx, i = 0):
        batch_item_embedding = self.emb(batch_item_idx)
        return (self.enc_sep[i](batch_user_embedding) * batch_item_embedding).sum(dim = 1)
    
    def forward_prob(self, batch_user_embedding, batch_item_idx, i = 0):
        return self.enc_sep[i](batch_user_embedding) @ self.emb(batch_item_idx).T
    
    def classify_rg(self, batch_user_embedding, st, ed, i = 0):
        return self.enc_sep[i](batch_user_embedding) @ self.emb.weight[st:ed].T
    
    def classify_all(self, batch_user_embedding, i = 0):
        return self.enc_sep[i](batch_user_embedding) @ self.emb.weight.T