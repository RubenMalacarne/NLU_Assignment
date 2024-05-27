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

class Parameters:

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    HID_SIZE = 350
    EMB_SIZE = 350
    LR = 0.0001
    CLIP = 5
    VOCAB_LEN = lambda x: len(x.word2id)
    N_EPOCHS = 100
    PATIENCE = 5
    
    BOOL_wait_tying = False
    BOOL_Variational_Dropout = False
    BOOL_ASGD = False

def pre_preparation_train():
    #preparation to train the dataset:  
    download_datase()    
    train_raw, dev_raw, test_raw= get_raw_dataset()    
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    train_loader =  DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader =    DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader =   DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    model = RNN_LSTM_VDWT(Parameters.EMB_SIZE, Parameters.HID_SIZE, Parameters.VOCAB_LEN(lang),pad_index=lang.word2id["<pad>"],weight_tyining=True, variational_dropout = False).to(Parameters.DEVICE)
    model.apply(init_weights)


    if Parameters.BOOL_ASGD:
        optimizer = NTASGD(model.parameters(), lr=Parameters.LR)
    else:
        optimizer = optim.SGD(model.parameters(),  lr=Parameters.LR)
        optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)

    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    ##la perplexity è calcolata in base alla somma dei criteri sopra diviso per il numero totale di sample
    #[(L1+L2)/number of total of samples]
    return train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval
    
#####dividere training loop ed evaulation loop
def train_part(train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval):
    print("now you are runnning the training part")
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
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


def eval(test_loader, criterion_eval, model):
    ppl, _ = eval_loop(test_loader, criterion_eval, model)
    print("- Test ppl:", ppl)
    
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

