import torch
import torch.utils.data as data
import math

import requests
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Add functions or classes used for data loading and preprocessing

# Downoad the dataset--------------------------------------------------------------------
def download_datase():
    
    filenames = ["ptb.test.txt","ptb.valid.txt","ptb.train.txt"]
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    subfolder_path = os.path.join(current_dir, 'dataset')
    # Crea la sottocartella se non esiste
    os.makedirs(subfolder_path, exist_ok=True)
    
    link = "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/"
    
    for filename in filenames:

        file_path = os.path.join(subfolder_path, filename)
        
        all_link = link + filename
        
        response = requests.get(all_link)

        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"the file  {filename} was successfully downloaded")
        else:
            print(f"Error during file downloaded. Status code: {response.status_code}")


# get_dataset--------------------------------------------------------
# Loading the corpus
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

def get_dataset_raw():
    train_raw = read_file("dataset/ptb.train.txt")
    valid_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    return train_raw, valid_raw, test_raw, vocab


def get_raw_dataset():
    try:
        train_raw = read_file("dataset/ptb.train.txt")
        dev_raw = read_file("dataset/ptb.valid.txt")
        test_raw = read_file("dataset/ptb.test.txt")
        #inserire caso in cui  il dataset non sia scaricato
        vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
        return train_raw, dev_raw, test_raw
    except:
        print ("dataset not found")

#Cosine similarity
def cosine_similarity(v, w):
    return np.dot(v,w)/(norm(v)*norm(w))


# This class computes and stores our vocab
# Word to ids and ids to word
#La funzione crea un vocabolario (mappatura parola-identificatore) a partire da un corpus di testo, includendo anche token speciali se specificati.
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

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