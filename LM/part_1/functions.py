# Here is where you define the architecture of your model using pytorch
from model import *
from utils import *
from functools import partial
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

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

def pre_preparation_train(EMB_SIZE,HID_SIZE,VOCAB_LEN,DEVICE,LR,SGD,ADAM,N_EPOCHS,PATIENCE,CLIP):
    #preparation to train the dataset:  
    download_datase()    
    train_raw, dev_raw, test_raw= get_raw_dataset()    
    
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    #get dataset
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset =   PennTreeBank(dev_raw, lang)
    test_dataset =  PennTreeBank(test_raw, lang)
    #get_dataloader
    train_loader =  DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader =    DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader =   DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    model = RNN_LSTM(EMB_SIZE, HID_SIZE, VOCAB_LEN(lang), pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)
    
    # choice the optimization
    if SGD:
        optimizer = optim.SGD(model.parameters(), lr=LR)
    elif ADAM: 
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
        torch.save(best_model, "bin/best_model.pt")
    else: print("Train non inizializzato")

def load_eval_model(DEVICE):
    #usato per caricare il modello e portarlo in evaluation
    model = torch.load('bin/best_model.pt', map_location=DEVICE)
    model.eval()
    return model

def eval_part(EVALUATION,test_loader, criterion_eval, model):
    if EVALUATION:
        ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print("Test ppl:", ppl)
    else: print("eval non inizializzato")
        


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

