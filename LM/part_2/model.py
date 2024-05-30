import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.utils.data as data

from utils import * 


class RNN_LSTM_VDWT(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,weight_tyining = False,variational_dropout = False , pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(RNN_LSTM_VDWT, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_size,padding_idx=pad_index)
        # after embedding layer
        self.embeding_dropout = nn.Dropout(emb_dropout)
        self.pad_token = pad_index
        self.lstm = nn.LSTM(input_size, hidden_size,n_layers)

        self.output = nn.Linear(hidden_size, vocab_size)

        if weight_tyining:
            # Weight tying: condivisione dei pesi tra il layer di output e l'embedding
            self.output.weight = self.embedding.weight  # Tying the weights

        if variational_dropout:

            self.out_dropout = VariationalDropout(out_dropout)
            self.emb_dropout = VariationalDropout(emb_dropout)


        #before last layer add out_droput
        self.output_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hidden_size, vocab_size)


    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.embeding_dropout(emb) #application of embedding dropout

        lstm_out, _ = self.lstm(emb)

        output = self.output_dropout(lstm_out) #droppout before last linear layer
        output = self.output(output).permute(0,2,1)
        return output


    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)

