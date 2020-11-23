import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import *

class Model(nn.Module):
    def __init__(self,config_path):
        super(Model,self).__init__()
        self.config = read_config(config_path)
        self.class_list = self.config["class_list"]
        self.vocab_path = self.config["vocab_path"]
        self.save_path = self.config["save_path"]
        self.filter_sizes = self.config["filter_sizes"]
        self.filter_num = self.config["filter_nums"]

        self.require_improvement = self.config["require_improvement"]
        self.n_vocab = self.config['vocabulary_size']
        self.embedding_dim = self.config['Embedding_dim']
        self.num_epochs = self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]
        self.pad_size = self.config["pad_size"]
        self.dropout = nn.Dropout(self.config["dropout"])
        self.device = torch.device('cuda')
        self.Embedding_pretrained = self.config["Embedding_pretrained"]

        self.convs = nn.ModuleList([nn.Conv2d(1,self.filter_num, (filter_size, self.embedding_dim)) for filter_size in self.filter_sizes]).to(self.device)
        self.fc = nn.Linear(self.filter_num*len(self.filter_sizes),len(self.class_list)).to(self.device)
        if self.Embedding_pretrained == True:
            self.Embedding = nn.Embedding.from_pretrained(self.config["Pretrain_Embedding_path"],freeze=False)
        else:
            self.Embedding = nn.Embedding(self.n_vocab,self.embedding_dim,padding_idx=self.n_vocab-1).to(self.device)

    def conv_and_pool(self,x, conv):
        x = F.relu(conv(x).to(self.device)).squeeze(3)
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.Embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
