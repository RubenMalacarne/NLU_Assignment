# import pycuda.driver as cuda
# import pycuda.autoinit

# DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu
# Import everything from functions.py file

from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # If you are using Colab, run these lines
    train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval= pre_preparation_train()
    train_part(train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval)
    
    #evaluation part
    eval_part(test_loader, criterion_eval, load_eval_model())