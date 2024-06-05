# import pycuda.driver as cuda
# import pycuda.autoinit

# DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu

# Import everything from functions.py file
from functions import *

class Parameters:
    #Parameter
    BOOL_WAIT_TYINING = False
    BOOL_VD = False
    BOOL_NTASGD = True
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    HID_SIZE = 350
    EMB_SIZE = 350
    
    LR = 0.95
    CLIP = 5
    
    VOCAB_LEN = lambda x: len(x.word2id)
    N_EPOCHS = 100
    PATIENCE = 5

    
    TRAINING = True
    EVALUATION = True

if __name__ == "__main__":
    
    train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval= pre_preparation_train(Parameters.BOOL_ASGD,
                                                                                                               Parameters.EMB_SIZE,
                                                                                                               Parameters.HID_SIZE,
                                                                                                               Parameters.VOCAB_LEN,
                                                                                                               Parameters.LR,
                                                                                                               Parameters.DEVICE)
    restemp=[]
    
    for r in range(0,5):
        print('Run', r+1)
        #INIT MODEL

        model.apply(init_weights)

        #OPTIMIZER
        optimizer =  optim.SGD(model.parameters(), lr=lr, weight_decay=1.2e-6) #same config as source code

        config = {
                    'n': 5,  # Non-monotone interval
                    'lr':lr,
                    'logs':[] #list to store validation losses
                    }
        train_part(Parameters.TRAINING,
                Parameters.N_EPOCHS,
                Parameters.DEVICE,
                Parameters.CLIP,
                Parameters.PATIENCE,
                Parameters.NTASGD,
                optimizer,
                model,
                train_loader,
                dev_loader,
                test_loader,
                criterion_eval,
                criterion_train,
                config=None)
    
    # eval_part (Parameters.EVALUATION,
    #            test_loader, 
    #            criterion_eval, 
    #            model)
    

torch.cuda.empty_cache()