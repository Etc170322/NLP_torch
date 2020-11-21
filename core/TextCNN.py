import torch
from torch import nn
import torch.functional as F
import numpy as np
from utils import *

class Model(nn.Module):
    def __init__(self,config):
        self.config = read_config(config)
        self.train_path = self.config["train_path"]
        self.test_path = self.config["test_path"]
        self.dev_path = self.config["dev_path"]
        self.class_list = self.config["class_list"]
        self.vocab_path = self.config["vocab_path"]
        self.model_path = self.config["model_path"]
        self.filter_sizes = self.config["filter_sizes"]
        self.filter_num = self.config["filter_num"]

        self.require_improvement = self.config["require_improvement"]
        self.n_vocab = 0
        self.num_epochs = self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]
        self.pad_size = self.config["pad_size"]
        self.dropout = self.config["dropout"]
        self.device = torch.device('cuda')
        self.Embedding_pretrained = self.config["Embedding_pretrained"]


    def embedding(self):
        if self.Embedding_pretrained is not None:
            self.Embedding = nn.Embedding.from_pretrained(self.config["Pretrain_Embedding_path"],freeze=False)
        else:
            self.Embedding = nn.Embedding(self.)