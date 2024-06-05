import torch
import torch.utils.data as data
import requests
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Add functions or classes used for data loading and preprocessing
def download_dataset():
    
    filenames = ["ptb.test.txt","ptb.valid.txt","ptb.train.txt"]
    current_file_path = os.path.abspath(__file__)
    current_dir =       os.path.dirname(current_file_path)
    subfolder_path =    os.path.join(current_dir, 'dataset')
    
    # create a subfolder if does not exist
    os.makedirs(subfolder_path, exist_ok=True)
    
    link = "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/"
    
    for filename in filenames:
        file_path = os.path.join(subfolder_path, filename)
        all_link = link + filename      
        response = requests.get(all_link)

        if response.status_code == 200:             #200 --> successful response
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"the file  {filename} was successfully downloaded")
        else:
            print(f"Error during file downloaded. Status code: {response.status_code}")

def read_file(path, eos_token="<eos>"):
    # add eos to indicate the "end of sentence"
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    vocab = {}
    i = 0
    for st in special_tokens:
        vocab[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in vocab:
                vocab[w] = i
                i += 1
    return vocab

def get_dataset_raw():

    train_raw = read_file("dataset/ptb.train.txt")
    valid_raw = read_file("dataset/ptb.valid.txt")
    test_raw =  read_file("dataset/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    return train_raw, valid_raw, test_raw, vocab

def get_raw_dataset():

    try:
        train_raw = read_file("dataset/ptb.train.txt")
        dev_raw =   read_file("dataset/ptb.valid.txt")
        test_raw =  read_file("dataset/ptb.test.txt")

        vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
        return train_raw, dev_raw, test_raw
    except:
        print ("ATTENTION: dataset not found, maybe not downloaded or you are not in the right folder to launch the program")

class Lang():

    # This class computes and stores our vocab
    # Word to ids and ids to word
    #La funzione crea un vocabolario (mappatura parola-identificatore) a partire da un corpus di testo, includendo anche token speciali se specificati.

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

    # Auxiliary method
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


def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

