from torch import nn
import torch.nn.functional as F
from utils import *
import torch

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.save_path = config.get("TextRNN_Att","save_path")
        self.class_list = config.get("Data","class_list").split(",")
        self.log_path = config.get("TextRNN_Att","log_path")
        self.hidden_size = config.getint("TextRNN_Att","hidden_size")
        self.num_layers = config.getint("TextRNN_Att","num_layers")
        self.bidirectional = config.getboolean("TextRNN_Att","bidirectional")
        self.batch_first = config.getboolean("TextRNN_Att","batch_first")
        self.n_vocab = config.getint("Embedding",'vocabulary_size')
        self.embedding_dim = config.getint("Embedding",'Embedding_dim')
        self.dropout = config.getfloat("TextRNN_Att","dropout")
        self.learning_rate = config.getfloat("TextRNN_Att","learning_rate")
        self.epoches = config.getint("TextRNN_Att","epoches")
        self.require_improvement = config.getint("TextRNN_Att","require_improvement")
        self.device = torch.device(config.get("Data","device"))
        self.Embedding_pretrained = config.getboolean("Embedding","Embedding_pretrained")

        if self.Embedding_pretrained == True:
            self.Embedding = nn.Embedding.from_pretrained(config.get("TextRNN_Att","Pretrain_Embedding_path"),freeze=False)
        else:
            self.Embedding = nn.Embedding(self.n_vocab+2,self.embedding_dim,padding_idx=self.n_vocab-1).to(self.device)

        self.LSTM =nn.LSTM(self.embedding_dim,self.hidden_size,self.num_layers,
                           bidirectional=self.bidirectional,batch_first=self.batch_first,dropout=self.dropout).to(self.device)
        self.tanh1 = nn.Tanh().to(self.device)
        self.w = nn.Parameter(torch.zeros(self.hidden_size*2)).to(self.device)
        self.tanh2 = nn.Tanh().to(self.device)

        self.fc = nn.Linear(self.hidden_size*2,len(self.class_list)).to(self.device)

    def forward(self, x):
        out = self.Embedding(x[0])
        out,_ = self.LSTM(out)
        out = self.tanh1(out)
        alpha = F.softmax(torch.matmul(out,self.w),dim=1).to(self.device).unsqueeze(-1)
        out = out*alpha
        out = torch.sum(out,dim=1)
        out = self.tanh2(out)
        out = self.fc(out)
        return out