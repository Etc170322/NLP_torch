from torch import nn
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.save_path = config.get("TextRNN","save_path")
        self.class_list = config.get("Data","class_list").split(",")
        self.log_path = config.get("TextRNN","log_path")
        self.hidden_size = config.getint("TextRNN","hidden_size")
        self.num_layers = config.getint("TextRNN","num_layers")
        self.bidirectional = config.getboolean("TextRNN","bidirectional")
        self.batch_first = config.getboolean("TextRNN","batch_first")
        self.n_vocab = config.getint("Embedding",'vocabulary_size')
        self.embedding_dim = config.getint("Embedding",'Embedding_dim')
        self.dropout = config.getfloat("TextRNN","dropout")
        self.learning_rate = config.getfloat("TextRNN","learning_rate")
        self.epoches = config.getint("TextRNN","epoches")
        self.require_improvement = config.getint("TextRNN","require_improvement")
        self.device = torch.device(config.get("Data","device"))
        self.Embedding_pretrained = config.getboolean("Embedding","Embedding_pretrained")

        if self.Embedding_pretrained == True:
            self.Embedding = nn.Embedding.from_pretrained(config.get("TextRNN","Pretrain_Embedding_path"),freeze=False)
        else:
            self.Embedding = nn.Embedding(self.n_vocab+2,self.embedding_dim,padding_idx=self.n_vocab-1).to(self.device)

        self.LSTM =nn.LSTM(self.embedding_dim,self.hidden_size,self.num_layers,
                           bidirectional=self.bidirectional,batch_first=self.batch_first,dropout=self.dropout).to(self.device)
        self.fc = nn.Linear(self.hidden_size*2,len(self.class_list)).to(self.device)

    def forward(self, x):
        out = self.Embedding(x[0])
        out,_ = self.LSTM(out)
        out = self.fc(out[:,-1,:])
        return out