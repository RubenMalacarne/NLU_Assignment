# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from torch.utils.data import DataLoader
import torch.optim as optim
class Parameters:

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    HID_SIZE = 350
    EMB_SIZE = 350
    LR = 0.0001
    CLIP = 5
    VOCAB_LEN = lambda x: len(x.word2id)
    N_EPOCHS = 10
    PATIENCE = 3
    BIDIRECTIONAL = False
    DRPOUT = True
    CRITERSION_SLOTS =nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    CRITERSION_INTENTS = nn.CrossEntropyLoss()  # Because we do not have the pad token


final_intent_dict = {}

final_slot_dict = {}

parameter_sets = [
    {"BIDIRECTIONAL": False, "DROPOUT": False,"description": "Original"},
    {"BIDIRECTIONAL": True,  "DROPOUT": False,"description": "Bidirectional"},
    {"BIDIRECTIONAL": True,  "DROPOUT": True, "description": "Bidirectional"}
]

if __name__ == "__main__":


    for sample_params  in parameter_sets:
      params = {}
      for key, value in sample_params.items():
        params[key] = value

      intent2id,slot2id,w2id,train_raw,dev_raw,test_raw,lang =pre_preparation_train()
      # Create our datasets
      train_dataset = IntentsAndSlots(train_raw, lang)
      dev_dataset = IntentsAndSlots(dev_raw, lang)
      test_dataset = IntentsAndSlots(test_raw, lang)

      # Dataloader instantiations
      train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
      dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
      test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

      out_slot = len(lang.slot2id)
      out_int = len(lang.intent2id)
      vocab_len = len(lang.word2id)

      model = ModelIAS(params["BIDIRECTIONAL"],params["DROPOUT"],Parameters.HID_SIZE, out_slot, out_int, Parameters.EMB_SIZE, vocab_len,pad_index=PAD_TOKEN).to(device)
      model.apply(init_weights)

      optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)

      sampled_epochs,losses_train,losses_dev,results_slot,results_intent,best_model = train_part(Parameters.PATIENCE,Parameters.N_EPOCHS,Parameters.CLIP,dev_loader,train_loader,test_loader,lang,optimizer,Parameters.CRITERSION_SLOTS,Parameters.CRITERSION_INTENTS,model)

      save_model(best_model, sample_params["description"])


      final_intent_dict [sample_params["description"]]  = results_intent
      final_slot_dict[sample_params["description"]]     = results_slot


      torch.cuda.empty_cache()
      torch.cuda.empty_cache()
            
      for sample_params  in parameter_sets:
          #evaluation part
          eval_part(Parameters.CRITERSION_SLOTS,
                Parameters.CRITERSION_INTENTS,
                load_eval_model(Parameters.DEVICE,sample_params["description"]),
                test_loader,lang)
