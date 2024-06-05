
import os
import copy
import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from model import *
from utils import *

from conll import evaluate

device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side



def pre_preparation_train():


          #Wrtite the code to load the datasets and to run your functions
    # Print the results
    portion = 0.10
    download_dataset()
    tmp_train_raw,test_raw = set_dataset()
    train_raw,dev_raw,test_raw = set_develop_dataset(portion, tmp_train_raw,test_raw)
    intent2id,slot2id,w2id = words_to_numbers_converter (train_raw,dev_raw,test_raw)
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                                # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels,
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    return intent2id,slot2id,w2id,train_raw,dev_raw,test_raw,lang

def train_part(PATIENCE,N_EPOCHS,CLIP,dev_loader,train_loader,test_loader,lang,optimizer,criterion_slots,criterion_intents,model):
    #pre_elaboration()
    losses_train = []
    losses_dev = []
    sampled_epochs = []

    best_f1 = 0

    best_model = None

    for x in tqdm(range(1,N_EPOCHS)):

        loss = train_loop(train_loader, optimizer, criterion_slots,
                        criterion_intents, model, clip=CLIP)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                patience = PATIENCE
                best_model = copy.deepcopy(model)
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                            criterion_intents, model, lang)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    results_slot = results_test['total']['f']
    results_intent =  intent_test['accuracy']
    return sampled_epochs,losses_train,losses_dev,results_slot,results_intent,best_model

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])

        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array

def eval_part(CRITERSION_SLOTS,CRITERSION_INTENTS, model,test_loader,lang):
    results_test, intent_test, _ = eval_loop(test_loader, CRITERSION_SLOTS, CRITERSION_INTENTS, model, lang)
    print("- Slot F1:", results_test["total"]["f"], end=" ")
    print("- Intent Accuracy:", intent_test["accuracy"])

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
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
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def save_model(best_model,name):
    print ("salvataggio modello...")

    if not os.path.exists("model_pt"):
      os.makedirs("model_pt")
    torch.save(best_model, "model_pt/"+name+".pth")

def load_eval_model(DEVICE,name):
    model_load = torch.load("model_pt/"+name+'.pth', map_location=DEVICE)
    model_load.eval()
    return model_load