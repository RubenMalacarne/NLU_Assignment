
from functions import *
class Parameters:
    #Parameter
    BOOL_WAIT_TYINING = True
    BOOL_VD = True
    BOOL_NTASGD = False

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    HID_SIZE = 256
    EMB_SIZE = 256

    LR = 0.001
    CLIP = 5

    VOCAB_LEN = lambda x: len(x.word2id)
    N_EPOCHS = 100
    PATIENCE = 10


    TRAINING = True
    EVALUATION = True


final_ppl_dict = {}

parameter_sets = [
    {"BOOL_WAIT_TYINING": False,"BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.001,"description": "LSTM"},
    {"BOOL_WAIT_TYINING": True, "BOOL_VD": False,"BOOL_NTASGD": False,"LR": 0.001,"description": "LSTM+WT"},
    {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": False,"LR": 0.001,"description": "LSTM+WT+VD"},
    {"BOOL_WAIT_TYINING": True, "BOOL_VD": True, "BOOL_NTASGD": True, "LR": 0.90,"description": "LSTM+VD+WT+NTASG"}
]
if __name__ == "__main__":


    for sample_params  in parameter_sets:
      params = {}
      for key, value in sample_params.items():
        params[key] = value

      train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval,lang= pre_preparation_train(params["BOOL_NTASGD"],
                                                                                                                Parameters.EMB_SIZE,
                                                                                                                Parameters.HID_SIZE,
                                                                                                                Parameters.VOCAB_LEN,
                                                                                                                params["LR"],
                                                                                                                Parameters.DEVICE,
                                                                                                                  params["BOOL_WAIT_TYINING"],
                                                                                                                  params["BOOL_VD"])
      restemp=[]


      model.apply(init_weights)
      if params["BOOL_NTASGD"]:
        #OPTIMIZER
        optimizer =  optim.SGD(model.parameters(), lr=params["LR"], weight_decay=1.2e-6) #same config as source code

        config = {
                    'n': 5,  # Non-monotone interval
                    'lr':params["LR"],
                    'logs':[] #list to store validation losses
                    }

        best_model,final_ppl_dict[sample_params["description"]] = train_part(
                                        Parameters.TRAINING,
                                        Parameters.N_EPOCHS,
                                        Parameters.DEVICE,
                                        Parameters.CLIP,
                                        Parameters.PATIENCE,
                                        params["BOOL_NTASGD"],
                                        optimizer,
                                        model,
                                        train_loader,
                                        dev_loader,
                                        test_loader,
                                        criterion_eval,
                                        criterion_train,
                                        config=config)
      else:
        best_model,final_ppl_dict[sample_params["description"]] = train_part(
                                        Parameters.TRAINING,
                                        Parameters.N_EPOCHS,
                                        Parameters.DEVICE,
                                        Parameters.CLIP,
                                        Parameters.PATIENCE,
                                        params["BOOL_NTASGD"],
                                        optimizer,
                                        model,
                                        train_loader,
                                        dev_loader,
                                        test_loader,
                                        criterion_eval,
                                        criterion_train)

      save_model(best_model,sample_params["description"])
      torch.cuda.empty_cache()

      for sample_params  in parameter_sets:
          #evaluation part
          eval_part(Parameters.EVALUATION,
                    test_loader,
                    criterion_eval,
                    load_eval_model(Parameters.DEVICE,sample_params["description"]))



      #take key and value final_ppl_dict:
      descriptions = list(final_ppl_dict.keys())
      final_perplexities = list(final_ppl_dict.values())
      # Plotting

      plt.errorbar(descriptions, final_perplexities, marker='o', linestyle='-', color='b', capsize=5)
      plt.title('Perplexity Values for Different Models')
      plt.xlabel('Models')
      plt.ylabel('Perplexity')
      plt.grid(True)
      # Set DPI (dots per inch)
      plt.figure(dpi=100)

      plt.show()
