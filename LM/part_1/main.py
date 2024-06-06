from functions import *
# plot graph


import matplotlib.pyplot as plt

class Parameters:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Hyperparameter
    #optimizer
    SGD = False
    ADAM = False
    RNN = False

    #during pre_preparation_train
    VOCAB_LEN = lambda x: len(x.word2id)

    #during training part
    CLIP = 5
    N_EPOCHS = 100

    HID_SIZE = 350
    EMB_SIZE = 350
    LR = 0.00001

    PATIENCE = 5

    TRAINING = True
    EVALUATION = True


final_ppl_dict = {}
parameter_sets = [
    {"SGD": True,  "ADAM": False,"LR": 0.3,"RNN": True, "description": "RNN_SGD"},
    {"SGD": False, "ADAM": True, "LR": 0.001,   "RNN": True, "description": "RNN_ADAM"},
    {"SGD": True, "ADAM": False, "LR": 0.8,"RNN": False,"description": "LSTM_SGD"},
    {"SGD": False, "ADAM": True, "LR": 0.001,"RNN": False,"description": "LSTM_ADAM"}
]


if __name__ == "__main__":


    for sample_params  in parameter_sets:
      params = {}
      for key, value in sample_params.items():
        params[key] = value

      train_loader,dev_loader,test_loader,model,optimizer,criterion_train,criterion_eval= pre_preparation_train(params["RNN"],
                                            Parameters.EMB_SIZE,
                                            Parameters.HID_SIZE,
                                            Parameters.VOCAB_LEN,
                                            Parameters.DEVICE,
                                            params["LR"],
                                            params["SGD"],
                                            params["ADAM"],
                                            Parameters.N_EPOCHS,
                                            Parameters.PATIENCE,
                                            Parameters.CLIP)

      final_ppl_dict[sample_params["description"]],best_model=train_part(Parameters.TRAINING,
                                                              Parameters.PATIENCE,
                                                              Parameters.N_EPOCHS,
                                                              Parameters.CLIP,
                                                              Parameters.DEVICE,
                                                              train_loader,
                                                              dev_loader,
                                                              test_loader,
                                                              model,
                                                              optimizer,
                                                              criterion_train,
                                                              criterion_eval)
      
      save_model(best_model,sample_params["description"])

      torch.cuda.empty_cache()

      #take key and value final_ppl_dict:
      descriptions = list(final_ppl_dict.keys())
      final_perplexities = list(final_ppl_dict.values())

      for sample_params  in parameter_sets:
          eval_part(Parameters.EVALUATION,
                    test_loader, 
                    criterion_eval, 
                    load_eval_model(Parameters.DEVICE,sample_params["description"]))

      # Plotting

      plt.errorbar(descriptions, final_perplexities, marker='o', linestyle='-', color='b', capsize=5)
      plt.title('Perplexity Values for Different Models')
      plt.xlabel('Models')
      plt.ylabel('Perplexity')
      plt.grid(True)
      # Set DPI (dots per inch)
      plt.figure(dpi=100)

      plt.show()