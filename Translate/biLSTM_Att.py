from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.device = config.device
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.Embedding_pretrained = config.Embedding_pretrained
        self.n_vocab = config.n_vocab

        if self.Embedding_pretrained == True:
            self.Embedding = nn.Embedding.from_pretrained(config.get("TextRNN_Att", "Pretrain_Embedding_path"),
                                                          freeze=False)
        else:
            self.Embedding = nn.Embedding(self.n_vocab+2,self.embedding_dim,padding_idx=self.n_vocab).to(self.device)
        self.LSTM = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout=self.dropout).to(self.device)

    def forward(self, input_x):
        embedded = self.Embedding(input_x)
        outputs,hidden = self.LSTM(embedded)
        outputs = (outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:])
        return outputs,hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

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

        self.encoder = Encoder(self.config)
        self.decoder = Encoder(self.config)

    def forward(self, input, teacher_forcing_ratio=0.5):
        src,trg = input
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs