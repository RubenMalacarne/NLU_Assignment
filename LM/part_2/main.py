# import pycuda.driver as cuda
# import pycuda.autoinit

# DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu

# Import everything from functions.py file
from functions import *


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # If you are using Colab, run these lines
    
    #nella seconda parte sistemare la parte dove si vuole fare true false di tutti e tre i tipi di valori
    train_loader,dev_loader,test_loader, model,optimizer,criterion_train,criterion_eval= pre_preparation_train()
    
    if Parameters.TRAINING:
        print("now you are runnning the training part")
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,Parameters.N_EPOCHS))

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, Parameters.CLIP)
            if epoch % 1 ==0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1

                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean        
        
        best_model.to(Parameters.DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)
    
        torch.save(best_model, "bin/best_model.pt")

    
    if Parameters.EVALUATION:    
        print("now you are Starting in the training part")
        ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print("Test ppl:", ppl)
    
