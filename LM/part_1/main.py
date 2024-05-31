# import pycuda.driver as cuda
# import pycuda.autoinit

# DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu
# Import everything from functions.py file

from functions import *

class Parameters:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    #Hyperparameter
    #optimizer
    SGD = False
    ADAM = True
    #during pre_preparation_train
    VOCAB_LEN = lambda x: len(x.word2id)
    #during training part
    CLIP = 5
    N_EPOCHS = 100
    
    HID_SIZE = 200
    EMB_SIZE = 300
    LR = 0.0001
    
    PATIENCE = 5
    
    TRAINING = True
    EVALUATION = True

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # If you are using Colab, run these lines
    train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval= pre_preparation_train(Parameters.EMB_SIZE,
                                                                                                               Parameters.HID_SIZE,
                                                                                                               Parameters.VOCAB_LEN,
                                                                                                               Parameters.DEVICE,
                                                                                                               Parameters.LR,
                                                                                                               Parameters.SGD,
                                                                                                               Parameters.ADAM,
                                                                                                               Parameters.N_EPOCHS,
                                                                                                               Parameters.PATIENCE,
                                                                                                               Parameters.CLIP)
    train_part(Parameters.TRAINING,
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
    
    #evaluation part
    eval_part(EVALUATION,test_loader, criterion_eval, load_eval_model(Parameters.DEVICE))