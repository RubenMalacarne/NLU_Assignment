from model import *
from utils import *

from functools import partial
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def pre_preparation_train(BOOL_NTASGD,EMB_SIZE,HID_SIZE,VOCAB_LEN,LR,DEVICE,BOOL_WAIT_TYINING,BOOL_VD):

    #preparation to train the dataset:
    download_dataset()
    train_raw, dev_raw, test_raw= get_raw_dataset()

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)


    train_loader =  DataLoader(train_dataset, batch_size=16, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader =    DataLoader(dev_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader =   DataLoader(test_dataset, batch_size=512, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    vocab_len = len(lang.word2id)

    model = MODEL_LSTM(EMB_SIZE, HID_SIZE, vocab_len,BOOL_WAIT_TYINING,BOOL_VD,pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)


    if BOOL_NTASGD:
        optimizer = NTASGD(model.parameters(), lr=LR)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
        #optimizer = optim.SGD(model.parameters(),  lr=Parameters.LR)
        #optimizer =  optim.SGD(model.parameters(), lr=LR, weight_decay=1.2e-6) #same config as source code


    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    vocab_len = len(lang.word2id)

    ##la perplexity è calcolata in base alla somma dei criteri sopra diviso per il numero totale di sample
    #[(L1+L2)/number of total of samples]
    return train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval,lang


def train_loop_NTASGD(data, optimizer, criterion, model, config, dev_loader, eval_criterion, clip=5):
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

def train_part_NASGD(N_EPOCHS,CLIP,PATIENCE,BOOL_NASGD,optimizer, criterion_train, criterion_eval,model,dev_loader, train_loader, config):
    pOg= copy.deepcopy(patience)
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,N_EPOCHS))
    for epoch in pbar:
        if not BOOL_NASGD:
            loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)
        else:
            loss,optimizer = train_loop_NTASGD(train_loader, optimizer, criterion_train, model, config, dev_loader, criterion_eval)

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
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print("Test ppl: ", final_ppl)
    return best_model


def train_part(TRAINING,N_EPOCHS,DEVICE,CLIP,PATIENCE,BOOL_NTASGD,optimizer,
               model,train_loader,dev_loader,test_loader,criterion_eval,
               criterion_train,config=None):
    if TRAINING:
      pOg= copy.deepcopy(PATIENCE)
      losses_train = []
      losses_dev = []
      sampled_epochs = []
      best_ppl = math.inf
      best_model = None
      pbar = tqdm(range(1,N_EPOCHS))
      for epoch in pbar:
          if not BOOL_NTASGD:
              loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)
          else:
              loss,optimizer = train_loop_NTASGD(train_loader, optimizer, criterion_train, model, config, dev_loader, criterion_eval)

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
      final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
      print("Test ppl: ", final_ppl)
      torch.save(best_model, "best_model.pt")

      return best_model, final_ppl

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



def save_model(best_model,name):
    print ("salvataggio modello...")

    if not os.path.exists("model_pt"):
      os.makedirs("model_pt")
    torch.save(best_model, "model_pt/"+name+".pt")

def load_eval_model(DEVICE,name):
    model = torch.load("model_pt/"+name+'.pt', map_location=DEVICE)
    model.eval()
    return model
