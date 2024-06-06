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

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        dataset = []
        with open(path) as f:
            dataset = json.loads(f.read())
        return dataset
    tmp_train_raw = load_data(os.path.join('dataset','train.json'))
    test_raw = load_data(os.path.join('dataset','test.json'))
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

    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    y_test = [x['intent'] for x in test_raw]

    return train_raw,dev_raw,test_raw

def words_to_numbers_converter(train_raw,dev_raw,test_raw):
    PAD_TOKEN = 0
    w2id = {'pad':PAD_TOKEN}
    slot2id = {'pad':PAD_TOKEN}
    intent2id = {}

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
            intent2id[example['intent']] = len (intent2id)

    for example in test_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
             slot2id[slot] = len(slot2id)

        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len (intent2id)

    sent = 'I wanna a flight from Toronto to Kuala Lumpur'

    print('# Vocab:', len(w2id)-2)
    print('# Slots:', len(slot2id)-1)
    print('# Intent:', len(intent2id))

    return intent2id,slot2id,w2id

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
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

    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)

    new_item["utterances"] = src_utt #is a 2 dimenszional vector batch size and maximum number the tokens in our batches
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

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
