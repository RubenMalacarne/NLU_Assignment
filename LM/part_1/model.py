import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.utils.data as data

# RNN Elman version
# We are not going to use this since for efficiently purposes it's better to use the RNN layer provided by pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

# RNN Elman version
# We are not going to use this since for efficiently purposes it's better to use the RNN layer provided by pytorch

class RNN_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(RNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size,padding_idx=pad_index)
        # after embedding layer
        self.embeding_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size,n_layers,bidirectional=False)
        #self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)
        self.pad_token = pad_index
        #before last layer add out_droput
        self.output_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hidden_size, vocab_size)



    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.embeding_dropout(emb) #application of embedding dropout
        rnn_out, _ = self.lstm(emb)
        output = self.output(rnn_out).permute(0,2,1)
        output = self.output_dropout(output) #droppout before last linear layer
        return output

    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        # Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(LM.cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)
    


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01) 
