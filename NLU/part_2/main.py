# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
from torch.utils.data import DataLoader
import torch.optim as optim


PAD_TOKEN = 0
if __name__ == "__main__":


    train_raw,dev_raw,test_raw,lang =pre_preparation_train()
    # Create our datasets
    train_dataset, dev_loader, test_dataset, lang =get_dataset(train_raw, dev_raw, test_raw)

    # Dataloader instantiations
    train_loader =  DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn,  shuffle=True)
    dev_loader =    DataLoader(dev_loader, batch_size=32, collate_fn=collate_fn)
    test_loader =   DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    out_slot =  len(lang.slot2id)
    out_int =   len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = BertCstm(Parameters.HID_SIZE, out_slot, out_int, Parameters.EMB_SIZE, vocab_len,pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)

    sampled_epochs,losses_train,losses_dev,best_model = train_part(Parameters.N_EPOCHS,Parameters.CLIP,Parameters.PATIENCE,dev_loader,train_loader,test_loader,lang,optimizer,Parameters.CRITERSION_SLOTS,Parameters.CRITERSION_INTENTS,model)
    
    save_model(best_model, "best_model")
    
    eval_part(Parameters.CRITERSION_SLOTS,Parameters.CRITERSION_INTENTS, load_eval_model(Parameters.DEVICE,"best_model"),test_loader,lang)



