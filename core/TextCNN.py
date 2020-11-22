import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import *

class Model(nn.Module):
    def __init__(self,config):
        self.config = read_config(config)
        self.class_list = self.config["class_list"]
        self.vocab_path = self.config["vocab_path"]
        self.model_path = self.config["model_path"]
        self.filter_sizes = self.config["filter_sizes"]
        self.filter_num = self.config["filter_num"]

        self.require_improvement = self.config["require_improvement"]
        self.n_vocab = self.config['vocabulary_size']
        self.embedding_dim = self.config['Embedding_dim']
        self.num_epochs = self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]
        self.pad_size = self.config["pad_size"]
        self.dropout = nn.Dropout(self.config["dropout"])
        self.device = torch.device('cuda')
        self.Embedding_pretrained = self.config["Embedding_pretrained"]

        self.convs = nn.ModuleList([nn.Conv2d(1,self.filter_num, self.embedding_dim, filter_size) for filter_size in self.filter_sizes])
        self.fc = nn.Linear(self.filter_num*len(self.filter_sizes),len(self.class_list))

    def embedding(self):
        if self.Embedding_pretrained is not None:
            self.Embedding = nn.Embedding.from_pretrained(self.config["Pretrain_Embedding_path"],freeze=False)
        else:
            self.Embedding = nn.Embedding(self.n_vocab,self.embedding_dim,padding_idx=self.n_vocab-1)

    def conv_and_pool(self,x, conv):
        x = F.relu(conv(x)).sequeeze(3)
        x = F.max_pool1d(x,x.size(2)).sequeeze(2)
        return x

    def forward(self, x):
        out = self.Embedding(x[0])
        out = out.unsequeeze(1)
        out = torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
