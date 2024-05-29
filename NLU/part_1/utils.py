# Add functions or classes used for data loading and preprocessing
# Add functions or classes used for data loading and preprocessing
import os
import requests
import json
from pprint import pprint
from collections import Counter

import torch.utils.data as data
import torch
from sklearn.model_selection import train_test_split

PAD_TOKEN =0
# Downoad the dataset--------------------------------------------------------------------
def download_dataset():
    
    filenames = ["test.json","train.json","conll.json"]
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    subfolder_path = os.path.join(current_dir, 'dataset/ATIS')
    
    # Crea la sottocartella se non esiste
    os.makedirs(subfolder_path, exist_ok=True)
    
    link = "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/"
    
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

def set_dataset():
    def load_data(path):
        '''
            input: path/to/data
            output: json
        '''
        dataset = []
        with open(path) as f:
            dataset = json.loads(f.read())
        return dataset
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))

    return tmp_train_raw,test_raw

def set_develop_dataset(portion,tmp_train_raw,test_raw):
    
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)
    
    labels = []
    inputs = []
    mini_train = []
    
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    #now do a trian end test split
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    print("mini train",mini_train)
    train_raw = X_train
    dev_raw = X_dev
    
    
    y_test = [x['intent'] for x in test_raw]
    
    # Intent distributions
    #print('Train:')
    pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    #print('Dev:'),
    pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    #print('Test:')
    pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    print('='*89)
    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))
    
    return train_raw,dev_raw,test_raw

def words_to_numbers_converter(train_raw,dev_raw,test_raw):
    PAD_TOKEN = 0
    w2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
    slot2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
    intent2id = {}

    # Map the words only from the train set
    # Map slot and intent labels of train, dev and test set. 'unk' is not needed.
    for example in train_raw:
        for w in example['utterance'].split():
                if w not in w2id:
                    w2id[w] = len(w2id)
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    for example in dev_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)

        if example['intent'] not in intent2id:
            intent2id[exmaple['intent']] = len (intent2id)

    for example in test_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
             slot2id[slot] = len(slot2id)

        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len (intent2id)

    sent = 'I wanna a flight from Toronto to Kuala Lumpur'


    print('# Vocab:', len(w2id)-2) # we remove pad and unk from the count
    print('# Slots:', len(slot2id)-1)
    print('# Intent:', len(intent2id))
    
    return intent2id,slot2id,w2id



class Lang():
    def __init__(self, words, intents, slots, cutoff=0):

        print (words)
        print (slots)
        print (intents)

        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)

        print (self.word2id)
        print (self.slot2id)
        print (self.intent2id)
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

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    #inizializzano i dataset da creare e si popolano con i valori presenti in "dataset"
    #usando la funzione mapping seq e lab si mappano
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances) #return the number of end we have in a dataset

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        #intent and slot are the output
        return sample

    # Auxiliary methods
    # per ogni etichetta(data) prensente nel dizionario (mapper) controlla se l'etichetta è presente nel dizionario di mapping
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    #data corrisponde ad una lista di sequenze per ogni sequenze , si divide la sequenza in parole e mappa ogni parola
    #al suo indicatore corrispondente utilizzando il dizionario di mapping
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

