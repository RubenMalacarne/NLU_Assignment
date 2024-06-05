# Add the class of your model only
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

def pre_preparation_train(BOOL_NTASGD,EMB_SIZE,HID_SIZE,VOCAB_LEN,LR,DEVICE):
    #preparation to train the dataset:  
    download_dataset()    
    train_raw, dev_raw, test_raw= get_raw_dataset()    
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    train_loader =  DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader =    DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader =   DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    model = MODEL_LSTM(EMB_SIZE, HID_SIZE, VOCAB_LEN(lang),pad_index=lang.word2id["<pad>"],weight_tyining=True, variational_dropout = False,dropprob=0.2).to(DEVICE)
    model.apply(init_weights)


    # if BOOL_NTASGD:
    #     optimizer = NTASGD(model.parameters(), lr=LR)
    # else:
    #     #optimizer = optim.Adam(model.parameters(), lr=LR)
    #     #optimizer = optim.SGD(model.parameters(),  lr=Parameters.LR)
    #     optimizer =  optim.SGD(model.parameters(), lr=LR, weight_decay=1.2e-6) #same config as source code

    optimizer =  optim.SGD(model.parameters(), lr=LR, weight_decay=1.2e-6) #same config as source code

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    ##la perplexity è calcolata in base alla somma dei criteri sopra diviso per il numero totale di sample
    #[(L1+L2)/number of total of samples]
    return train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval

def train_loop_NTASGD(NTASGD,data, optimizer, criterion, model, config, dev_loader, eval_criterion, clip=5):
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
    if NTASGD:    
        #NT-ASGD
        #Simply change to ASGD when the NT condition is met
        _,val_loss = eval_loop(dev_loader, eval_criterion, model)
        if 't0' not in optimizer.param_groups[0] and (len(config['logs'])>config['n'] and val_loss > min(config['logs'][:-config['n']])):
                print('\nNT condition met, switching to ASGD', optimizer)
                optimizer = torch.optim.ASGD(model.parameters(), lr=config['lr'], t0=0, lambd=0., weight_decay=1.2e-6)
                print(optimizer)
        config['logs'].append(val_loss)

    return sum(loss_array)/sum(number_of_tokens),optimizer

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


def train_part(TRAINING,N_EPOCHS,DEVICE,CLIP,PATIENCE,NTASGD,optimizer,model,train_loader,dev_loader,test_loader,criterion_eval,criterion_train,config=None):
    if TRAINING:
        
        print("now you are runnning the training part")
        pOg= copy.deepcopy(PATIENCE)
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,N_EPOCHS))
        for epoch in pbar:
            
            if NTASGD:
                loss,optimizer = train_loop_NTASGD(train_loader, optimizer, criterion_train, model, config, dev_loader, criterion_eval)
            else: 
                loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)
            
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f, Best: %f" % (ppl_dev, best_ppl))
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = pOg
                else:
                    patience -= 1

                if patience <= 0: # Early stopping with patience
                    print('Patience Limit Reached!')
                    break # Not nice but it keeps the code clean

            best_model.to(DEVICE)
            final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
            print('Test ppl: ', final_ppl)


    else: print ("TRAINING NOT INITIALIZING")

    return best_model

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


def eval_part(EVALUATION,test_loader, criterion_eval, model):
    if EVALUATION:
        ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print("- Test ppl:", ppl)
    else: print ("evaluation not run")
    
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
