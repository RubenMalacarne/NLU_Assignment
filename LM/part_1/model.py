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
    def __init__(self, input_size, hidden_size, vocab_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(RNN_LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_size,padding_idx=pad_index)
        # after embedding layer
        self.embeding_dropout = nn.Dropout(emb_dropout)

        self.lstm = nn.LSTM(input_size, hidden_size,n_layers)
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

