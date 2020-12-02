from torch import nn
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.save_path = config.get("TextCNN","save_path")
        self.class_list = config.get("Data","class_list").split(",")
        self.log_path = config.get("TextCNN","log_path")
        self.filter_sizes = config.get("TextCNN","filter_sizes").split(",")
        self.filter_num = config.getint("TextCNN","filter_nums")
        self.n_vocab = config.getint("Embedding",'vocabulary_size')
        self.embedding_dim = config.getint("Embedding",'Embedding_dim')
        self.dropout = nn.Dropout(config.getfloat("TextCNN","dropout"))
        self.learning_rate = config.getfloat("TextCNN","learning_rate")
        self.epoches = config.getint("TextCNN","epoches")
        self.require_improvement = config.getint("TextCNN","require_improvement")
        self.device = torch.device('cuda')
        self.Embedding_pretrained = config.getboolean("Embedding","Embedding_pretrained")

        self.convs = nn.ModuleList([nn.Conv2d(1,self.filter_num, (int(filter_size), self.embedding_dim)) for filter_size in self.filter_sizes]).to(self.device)
        self.fc = nn.Linear(self.filter_num*len(self.filter_sizes),len(self.class_list)).to(self.device)
        if self.Embedding_pretrained == True:
            self.Embedding = nn.Embedding.from_pretrained(config.get("TextCNN","Pretrain_Embedding_path"),freeze=False)
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
