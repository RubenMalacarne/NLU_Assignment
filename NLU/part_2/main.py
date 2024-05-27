# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
if __name__ == "__main__":

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    
    download_dataset()
    
    tmp_train_raw,test_raw = set_dataset()
    
    train_raw,dev_raw,test_raw = set_develop_dataset(tmp_train_raw,test_raw)    
    words, intents, slots = get_lang (train_raw,dev_raw,test_raw)
    
   
    lang = Lang(words, intents, slots, cutoff=0)
    # Create our datasets
    train_dataset = Bert_Dataset(train_raw, lang)
    dev_dataset = Bert_Dataset(dev_raw, lang)
    test_dataset = Bert_Dataset(test_raw, lang)
    
        
    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
            
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    #output sizes
    out_int=len(intents)
    out_slot=len(lang.slot2id)
    
    #store the results in these
    slot_met, intent_met = [[],[],[]], [[],[],[]]

    model = BertCstm(Parameters.HID_SIZE, out_slot, out_int, Parameters.EMB_SIZE, vocab_len,pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    
    best_f1=0
    
    patience=Parameters.PAT
    
    for x in tqdm(range(1,Parameters.N_EPOCHS)):

        #TRAINING
        model.train()
        
        loss = train_loop(train_loader, train_dataset, Parameters.DEVICE, optimizer,
                            model,criterion_intents, criterion_slots,
                            x, Parameters.N_EPOCHS,lang,tokenizer,dev_loader,
                            patience,best_f1,Parameters.PAT)   
        
        if patience <=0:
            print('Patience limit reached!')
            break
    
    #TESTING
    #load best model for testing
    model = BertCstm(bert, Parameters.HID_SIZE, out_slot, out_int)
    model.to(Parameters.DEVICE)
    model.load_state_dict(torch.load('best.pth', map_location=Parameters.DEVICE))
    model.eval()
    slot_met, intent_met = evalModel(model, test_loader, criterion_slots, criterion_intents, lang, slot_met, intent_met, tokenizer) 
    
      
    #RESULTS
    printRes(slot_met, intent_met, 'Bert')
        
    # results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
    #                                         criterion_intents, model, lang)
    
    
    # print('Slot F1: ', results_test['total']['f'])
    # print('Intent Accuracy:', intent_test['accuracy'])

    # plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    # plt.title('Train and Dev Losses')
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.plot(sampled_epochs, losses_train, label='Train loss')
    # plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    # plt.legend()
    # plt.show()