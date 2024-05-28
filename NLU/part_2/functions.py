# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import os
import requests
import json
from pprint import pprint

from model import *
from utils import *

from functools import partial
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from   tqdm import tqdm
import copy

import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
import torch.utils.data as data

from torch.utils.data import DataLoader

import re


from conll import evaluate ##check the performance at chunck level
from sklearn.metrics import classification_report

import torch.optim as optim

import matplotlib.pyplot as plt
#paremetri

device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0


class Parameters:

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    RUNS = 5
    HID_SIZE = 350
    EMB_SIZE = 350
    LR = 0.0001
    CLIP = 5
    VOCAB_LEN = lambda x: len(x.word2id)
    N_EPOCHS = 200
    PATIENCE = 3
    PAT = 7
    

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        #Dizionari inversi che mappano indicatori univoci alle loro rispettive parole slot e intenti
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    #Questo metodo prende una lista di parole e restituisce un dizionario che mappa ogni parola al suo identificatore univoco.
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    #Crea un dizionario che mappa ogni etichetta al suo identificatore univoco.
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    

class Bert_Dataset (data.Dataset):
    """
    Custom dataset class for PyTorch DataLoader.

    Methods:
    - __init__: Initialize the dataset.
    - __len__: Get the length of the dataset.
    - __getitem__: Get an item from the dataset.
    - mapping_lab: Map labels to indices.
    - mapping_seq: Map sequences to indices.
    """

    def __init__(self, data, lang, tokenizer):
        #WHAT DO WE PUT HERE?
        #copy the data
        self.data = copy.deepcopy(data)
        #load the given tokenizer
        self.tokenizer = tokenizer
        
        self.intents = []
        self.slots = []
        #get separate lists
        for x in self.data:
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])
        
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)


    def __getitem__(self, idx):
        #TAKE ONE ITEM FROM THE DATASET
        utt = self.data[idx]['utterance']
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        
        #fix o'clocl to clock
        utt = re.sub(r"(\w+)'(\w)", r"\1\2", utt)
        #utt = utt.replace("o'clock", 'clock')
        
        #Tokenizer splits ' and . 
        txt = utt.split(' ')
        for i in range(0,len(txt)):
            if txt[i].endswith('.'):
                txt[i] = txt[i][:-1]
        utt = ' '.join(txt)
        #tokens = []
        
        joint = self.tokenizer.encode(utt)
        tkns = self.tokenizer.convert_ids_to_tokens(joint)
        
        split = []
        for i,t in enumerate(tkns):
            if not t.startswith('##') and not t.startswith("'"):     
                split.append(t)
                
        split = self.tokenizer.convert_tokens_to_ids(split)

        split = torch.tensor(split)
        slots = torch.cat([torch.tensor([PAD_TOKEN]),slots,torch.tensor([PAD_TOKEN])]) #pad slot
        
        #just in case code fails
        if len(split) != len(slots):
            print('ERROR', split, slots, utt)

        return {
            'utt': utt,
            #'decoded': decode,
            'utterance': split, #encoding_text['input_ids'].flatten(),
            'text_attention_mask': torch.ones_like(split),#encoding_text['attention_mask'].flatten(),
            'slots': slots,
            'intent': intent,
            'pad':0
        }

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def collate_fn(data):
    """
    Collate function for the DataLoader.

    Args:
    - data: List of items from the dataset.

    Returns:
    Dictionary containing the collated data.
    """
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e., 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    text_attention_mask, _ = merge(new_item["text_attention_mask"])
    
    src_utt = src_utt.to(device)  # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    text_attention_mask = text_attention_mask.to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["text_attention_mask"] = text_attention_mask
    
    return new_item

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
                    
               
def train_loop(train_dl,train_ds,device,optimizer,model,criterion_intents,
               criterion_slots,epoch,n_epochs,lang,tokenizer,dev_dl,patience,
               best_f1,ogP):
    """
    Training loop for the model.

    Args:
    - train_dl: Training data loader.
    - train_ds: Training dataset.
    - device: Device on which to train the model.
    - optimizer: Model optimizer.
    - model: The model to be trained.
    - criterion_intents: Criterion for intent classification.
    - criterion_slots: Criterion for slot classification.
    - epoch: Current epoch number.
    - n_epochs: Total number of epochs.
    - lang: Vocabulary and label mappings (output of Lang() class).
    - tokenizer: Tokenizer for text data.
    - dev_dl: Development data loader for evaluation.
    - patience: Patience parameter for early stopping.
    - best_f1: Best F1 score achieved so far.

    Returns:
    updated patience and best F1 score.
    """
    total_loss=0
    for _, data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        
        utt = data['utterances'].to(device)
        attm = data['text_attention_mask']#.to(device)
        slot = data['y_slots'].to(device)
        intent = data['intents'].to(device)
    
        optimizer.zero_grad()
    
        predS, predI = model(utt,attm)
        
        lossS = criterion_slots(predS, slot)
        lossI = criterion_intents(predI, intent)

        # Backpropagation and optimization
        loss = lossS + lossI
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    results, report_intent, loss_array = eval_loop(dev_dl, criterion_slots,
                                                   criterion_intents, model,
                                                   lang, tokenizer)
    
    f1 = results['total']['f']
    #PATIENCE CODE
    if True:#epoch%2==0:
        
        if f1/total_loss > best_f1 and f1 > 0.90: #just to avoid entering here in the first epochs
            best_f1 = f1/total_loss
            torch.save(model.state_dict(), 'best.pth')
            patience=copy.deepcopy(ogP)
            print('Best E',epoch,patience)
        else:
            patience-=1
            
    print(f'Epoch [{epoch}/{n_epochs}], Loss: {total_loss / len(train_dl)}, F1: {f1}')
            
    return patience,best_f1
def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    """
    Evaluation loop for the model.

    Args:
    - data: Evaluation data.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - model: The model to be evaluated.
    - lang: Vocabulary and label mappings (output of Lang() class).
    - tokenizer: Tokenizer for text data.

    Returns:
    Tuple containing evaluation results, intent classification report, and loss array.
    """
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): 
        for sample in data:
            attm = sample['text_attention_mask']
            slots, intents = model(sample['utterances'], attm)
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                
                #Get the lenght of the slots w cls pad
                length = sample['slots_len'].tolist()[id_seq]                
                #remove the batch padding
                predSlot = seq[:length]
                #remove the cls padding
                predSlot = predSlot[1:-1]
                #turn into list
                predSlot = predSlot.tolist()
                goalSlot = sample['slots'][id_seq].tolist()[1:-1]
                
                #Get the slots in text
                txtpredSlot = [lang.id2slot[elem] for elem in predSlot]
                txtgoalSlot = [lang.id2slot[elem] for elem in goalSlot]
                #Get the utter in text
                txtutter = tokenizer.decode(sample['utterance'][id_seq][1:-1]).split(" ")
                #txtgoalUtter = sample['utt'][id_seq].split(" ")
                
                #Format targets as a list of tupples
                ref_slots.append([(txtutter[id_el], elem) for id_el, elem in enumerate(txtgoalSlot)])
                #Format predic as a list of tupples
                hyp_slots.append([(txtutter[id_el], elem) for id_el, elem in enumerate(txtpredSlot)])
                
                # #utt_ids = sample['utterance'][id_seq][:length].tolist()
                # gt_ids = sample['y_slots'][id_seq].tolist()
                # if sample['pad'][id_seq] != 0: #handle padding
                #     gt_ids = gt_ids[:-sample['pad'][id_seq]]
                #     seq = seq[:len(gt_ids)]
                # gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                # utterance = sample['utt'][id_seq].split() #[tokenizer.decode(elem) for elem in utt_ids][1:-1] #remove cls and sep #[lang.id2word[elem] for elem in utt_ids]
                # to_decode = seq[:length].tolist()
                # ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                # tmp_seq = []
                # for id_el, elem in enumerate(to_decode):
                #     if lang.id2slot[elem] == 'O':
                #         sl = 'O-O'
                #     else:
                #         sl = lang.id2slot[elem]
                #     tmp_seq.append((utterance[id_el], sl))
                # hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def evalModel(model, test_loader, criterion_slots, criterion_intents,lang,
              sltm,intm,tokenizer):
    """
    Evaluate the model on a test set.

    Args:
    - model: The model to be evaluated.
    - test_loader: Test data loader.
    - criterion_slots: Criterion for slot classification.
    - criterion_intents: Criterion for intent classification.
    - lang: Vocabulary and label mappings (output of Lang() class).
    - sltm: List to store slot-related metrics.
    - intm: List to store intent-related metrics.
    - tokenizer: Tokenizer for text data.

    Returns:
    Tuple containing updated slot-related metrics and intent-related metrics.
    """
    res_slt, res_int, _  = eval_loop(test_loader, criterion_slots, 
                                             criterion_intents, model, lang,tokenizer)

    sltm[0].append(res_slt['total']['f'])
    sltm[1].append(res_slt['total']['p'])
    sltm[2].append(res_slt['total']['r'])
    intm[0].append(res_int['weighted avg']['f1-score'])
    intm[1].append(res_int['weighted avg']['precision'])
    intm[2].append(res_int['weighted avg']['recall'])

    return sltm, intm

def printRes(sltm, intm, name):
    """
    Print evaluation results.

    Args:
    - sltm: Slot-related metrics.
    - intm: Intent-related metrics.
    - name: Name of the evaluation set.

    Returns:
    None
    """
    print('\n', name)
    n = ['F1','Precision','Recall']
    for i in range(0,3):
        smet = np.array(sltm[i])
        imet = np.array(intm[i])
        print(n[i]+':', 'S:', round(smet.mean(),4),'+-', round(smet.std(),4),
              'I:', round(imet.mean(),4),'+-', round(imet.std(),4))