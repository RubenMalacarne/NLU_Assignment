# Add functions or classes used for data loading and preprocessing
import os
import requests
import json
import torch

from collections import Counter
from sklearn.model_selection import train_test_split
import torch

import torch.utils.data as data

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    tmp_train_raw = load_data(os.path.join('dataset','train.json'))
    test_raw = load_data(os.path.join('dataset','test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))

    return tmp_train_raw,test_raw

def set_develop_dataset(tmp_train_raw,test_raw):

    portion = 0.10

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
    #now do a trian end test spli
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    print("mini train",mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw,dev_raw,test_raw

def get_lang(train_raw, dev_raw, test_raw):

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                                # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels,
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    return words, intents, slots


class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk="unk"):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id, encoding=True)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {"utterance": utt, "slots": slots, "intent": intent}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper, encoding=False):  # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            if encoding:
                tmp_seq.append(mapper["cls"])
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            if encoding:
                tmp_seq.append(mapper["sep"])
            res.append(tmp_seq)
        return res