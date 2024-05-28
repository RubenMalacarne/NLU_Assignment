# Here is where you define the architecture of your model using pytorch
from model import *
from utils import *
from functools import partial
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class Parameters:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    #Hyperparameter
    #optimizer
    SGD = True
    ADAM = False

    #during pre_preparation_train
    VOCAB_LEN = lambda x: len(x.word2id)
    
    #during training part
    CLIP = 5
    N_EPOCHS = 100
    
    HID_SIZE = 200
    EMB_SIZE = 300
    LR = 0.0001
    
    PATIENCE = 5


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

    new_item["source"] = source.to(Parameters.DEVICE)
    new_item["target"] = target.to(Parameters.DEVICE)
    new_item["number_tokens"] = sum(lengths)
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

def pre_preparation_train():
    #preparation to train the dataset:  
    download_datase()    
    train_raw, dev_raw, test_raw= get_raw_dataset()    
    
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset =   PennTreeBank(dev_raw, lang)
    test_dataset =  PennTreeBank(test_raw, lang)
    
    train_loader =  DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader =    DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader =   DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    model = RNN_LSTM(Parameters.EMB_SIZE, Parameters.HID_SIZE, Parameters.VOCAB_LEN(lang), pad_index=lang.word2id["<pad>"]).to(Parameters.DEVICE)
    model.apply(init_weights)
    
    # choice the optimization
    if Parameters.SGD:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif Parameters.ADAM: 
        optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    ##la perplexity è calcolata in base alla somma dei criteri sopra diviso per il numero totale di sample
    #[(L1+L2)/number of total of samples]
    
    return train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval    

def train_part(train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval):
    print("now you are runnning the training part")
    
    losses_train =  []
    losses_dev =    []
    sampled_epochs =[]
    
    best_ppl =  math.inf
    best_model = None
    pbar = tqdm(range(1,Parameters.N_EPOCHS))

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, Parameters.CLIP)
        if epoch % 1 ==0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean        
    
    best_model.to(Parameters.DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    torch.save(best_model, "bin/best_model.pt")

def load_eval_model():
    #usato per caricare il modello e portarlo in evaluation
    model = torch.load('bin/best_model.pt', map_location=Parameters.DEVIC)
    model.eval()
    return model

def eval_part(test_loader, criterion_eval, model):
    ppl, _ = eval_loop(test_loader, criterion_eval, model)
    print("Test ppl:", ppl)
    

