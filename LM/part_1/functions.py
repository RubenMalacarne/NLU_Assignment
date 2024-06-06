# Here is where you define the architecture of your model using pytorch
from functools import partial
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import torch
import torch.utils.data as data
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils import *
from model import *

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def pre_preparation_train(RNN,EMB_SIZE,HID_SIZE,VOCAB_LEN,DEVICE,LR,SGD,ADAM,N_EPOCHS,PATIENCE,CLIP):
    #preparation to train the dataset:
    download_dataset()
    train_raw, dev_raw, test_raw= get_raw_dataset()

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    #get dataset
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset =   PennTreeBank(dev_raw, lang)
    test_dataset =  PennTreeBank(test_raw, lang)
    #get_dataloader
    train_loader =  DataLoader(train_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader =    DataLoader(dev_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader =   DataLoader(test_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    model = RNN_LSTM(RNN,EMB_SIZE, HID_SIZE, VOCAB_LEN(lang), pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    # choice the optimization
    if SGD:
        optimizer = optim.SGD(model.parameters(), lr=LR)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    ##la perplexity è calcolata in base alla somma dei criteri sopra diviso per il numero totale di sample
    #[(L1+L2)/number of total of samples]

    return train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval

def train_part(TRAINING,PATIENCE,N_EPOCHS,CLIP,DEVICE,train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval):
    if TRAINING:
        print("now you are runnning the training part")

        losses_train =  []
        losses_dev =    []
        sampled_epochs =[]

        best_ppl =  math.inf
        best_model = None
        pbar = tqdm(range(1,N_EPOCHS))
        patience = PATIENCE

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)
            if epoch % 1 ==0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = PATIENCE
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        best_model.to(DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)

    else:
        print("Train non inizializzato")
        final_ppl = 0
    return final_ppl,best_model

def eval_part(EVALUATION,test_loader, criterion_eval, model):
    if EVALUATION:
        ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print("Test ppl:", ppl)
    else: print("eval non inizializzato")

def save_model(best_model,name):
    print ("salvataggio modello...")

    if not os.path.exists("model_pt"):
      os.makedirs("model_pt")
    torch.save(best_model, "model_pt/"+name+".pt")

def load_eval_model(DEVICE,name):
    model = torch.load("model_pt/"+name+'.pt', map_location=DEVICE)
    model.eval()
    return model
