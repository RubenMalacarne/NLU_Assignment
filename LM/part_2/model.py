import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.utils.data as data

from utils import * 


class MODEL_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,weight_tyining = False,variational_dropout = False , pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(MODEL_LSTM, self).__init__()

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



class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()   
        self.dropout = dropout

    def forward(self, x ):
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x

class NTASGD(optim.Optimizer):
    def __init__(self, params, lr=1, n=5):
        defaults = dict(lr=lr, n=n, t0=10e7, t=0, logs=[])
        super(NTASGD, self).__init__(params, defaults)

    def check(self, v):
        group = self.param_groups[0]
        #Training
        if group['t'] > group['n'] and v > min(group['logs'][:-group['n']]):
            group['t0'] = self.state[next(iter(group['params']))]['step']
            print("Non-monotonic condition is triggered!")
            return True
        group['logs'].append(v)
        group['t'] += 1

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

    def step(self):
        group = self.param_groups[0]
        for p in group['params']:
            grad = p.grad.data
            state = self.state[p]
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['mu'] = 1
                state['ax'] = torch.zeros_like(p.data)
            state['step'] += 1

            p.data.add_(other=grad, alpha=-group['lr'])
            # averaging
            if state['mu'] != 1:
                state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
            else:
                state['ax'].copy_(p.data)
            # update mu
            state['mu'] = 1 / max(1, state['step'] - group['t0'])