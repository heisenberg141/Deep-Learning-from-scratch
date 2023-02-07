import numpy as np


def main():
    input("PRESS Enter to run the code for double cross validation")
    double_cross_validation_runner()

    input("PRESS Enter to run the code for linear regression with SGD")
    linear_regression_sgd_runner()
    
    return

'''
_____________________Helper functions for Stochastic Gradient Descent________________________

'''

def linear_regression_sgd_runner():

    # Linera regression with SGD
    batch_sizes =   [512, 1024, 2048,4000]
    learning_rates = [0.001, 0.01,0.05,0.1]
    epochs = [1,5,10,15]
    alphas = [0.0001,0.001, 0.01,0.05]
    H = list()
    for i in range(len(batch_sizes)):
        for j in range(len(batch_sizes)):
            for k in range(len(batch_sizes)):
                for l in range(len(batch_sizes)):
                    H.append([int(batch_sizes[i]),learning_rates[j],int(epochs[k]),alphas[l]])
    
    
    # STEP 1 load the dataset.
    # n = 5k images in train
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")
    
    allIdxs = np.arange(len(X_tr))
    idxs = np.random.permutation(allIdxs)
    X_tr = X_tr[idxs]
    ytr = y_tr[idxs]
    
    #STEP2 getting the best hyper parameter config using Unregularized SGD
    # star_h = find_best_hps(X_tr, y_tr, H) 
    star_h = find_best_hps(X_tr, y_tr, H) 
    # star_h = [512, 0.001, 10, 0.05]
   
    model = train_linear_regression(X_tr, y_tr, star_h)

    test_fmse = test_model(X_te,y_te,model)

    print(f"(BEST) Test FMSE: {test_fmse}")

def test_model(X_te,y_te,model):
    print("FINAL TESTING on the test set")
    W,b = model
    testing_fmse=np.mean(np.square(np.dot(X_te, W) +  b*np.ones(y_te.shape) - y_te))/2
    return testing_fmse
        
def train_linear_regression(X_tr, y_tr, h):
    # batch size, learning rate, epochs, alpha
    batch_size, lr, epochs, alpha = h
    W = np.random.normal(0, 0.1,X_tr.shape[1])
    b = np.random.normal(0,0.1)
    print("TRAINING on the whole train set..")
    for epoch in range(epochs):
        mini_batches = generate_mini_batches(X_tr, y_tr,batch_size)
        for batch in mini_batches:
            x,y = batch
            y = y.reshape((y.shape[0],))
            x = x.T
                
            y_hat = x.T.dot(W) + np.ones(len(y))*b
                
            gradient_term_w = lr * x.dot(y_hat-y)/len(y)+ 2*alpha*W
            gradient_term_b = np.mean(y_hat-y)
            W-=gradient_term_w
            b-=gradient_term_b
    return W,b

def find_best_hps(X_tr,y_tr, H):
    allIdxs = np.arange(len(X_tr))
    idxs = np.random.permutation(allIdxs)
    X_tr = X_tr[idxs]
    ytr = y_tr[idxs]
    val_split = 0.2
    X_train, y_train, X_val, y_val = train_val_split(X_tr,y_tr,val_split)

    star_h = [0,0,0,0]
    best_loss = 1e6
    for h in H:
        # batch size, learning rate, epochs, alpha
        batch_size, lr, epochs, alpha = h
        w = np.random.normal(0, 0.1,X_train.shape[1])
        b = np.random.normal(0,0.1)
        print(f"Training with h = {h}. ")
        for epoch in range(epochs):
            mini_batches = generate_mini_batches(X_train, y_train,batch_size)
            for batch in mini_batches:
                x,y = batch
                y = y.reshape((y.shape[0],))
                x = x.T
                
                y_hat = x.T.dot(w) + np.ones(len(y))*b
                
                gradient_term_w = lr * x.dot(y_hat-y)/len(y)+ 2*alpha*w
                gradient_term_b = np.mean(y_hat-y)
                w-=gradient_term_w
                b-=gradient_term_b

        print("Validating")
        validation_fmse=np.mean(np.square(np.dot(X_val, w) +  b*np.ones(y_val.shape) - y_val))/2
        if validation_fmse<best_loss:
            best_loss = validation_fmse
            star_h = h
            print(f"New Best loss: {best_loss}")
    print("\n____________________________DONE___________________________________\n")
    print(f"BEST FMSE loss: {best_loss}\n h_star: {star_h}")
    return star_h
    
def generate_mini_batches(X, y, batch_size):
    
    mini_batches = []
    y = np.array([y])
    data = np.append(X,y.T,axis=1)
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches ):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    
    return mini_batches

def train_val_split(X_tr,y_tr,val_split): 
    all_idx = np.arange(len(X_tr))
    X_val = X_tr[:int(val_split*len(X_tr)),]
    y_val = y_tr[:int(val_split*len(X_tr)),]
    X_train = X_tr[int(val_split*len(X_tr)):,]
    y_train = y_tr[int(val_split*len(X_tr)):,]
    return X_train, y_train, X_val, y_val
     
'''
_____________________Helper functions for double cross validation________________________

'''

def double_cross_validation_runner():
    D = np.random.rand(32,32)
    print(D)     
    H = np.arange(4)
    k = 4
    doDoubleCrossValidation(D,k,H)

def testModel(model,D):
    return np.random.rand()

def trainModel(D,h):
    return "Model Training Successful"

def doCrossValidation (D, k, h):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    for fold in range(k):
        # Get all indexes for this fold
        testIdxs = idxs[fold,:]
        # Get all the other indexes
        trainIdxs = np.array(list(set(allIdxs) - set(testIdxs))).flatten()
        # Train the model on the training data
        model = trainModel(D[trainIdxs], h)
        # Test the model on the testing data
        accuracies.append(testModel(model, D[testIdxs]))
    print(f"History of accuracies for inner folds where h = {h} :", accuracies,"\nAverage: ", np.mean(accuracies),"\n")

    return np.mean(accuracies)

def get_best_hyperparameters(D_OutTrIdx, k, H):
    best_h = H[0]
    best_accuracy = 0
    all_accuracies = list()
    for h in H:
        average_accuracy = doCrossValidation(D_OutTrIdx, k, h)
        all_accuracies.append((h,average_accuracy))
        if average_accuracy>best_accuracy:
            best_h = h
            best_accuracy = average_accuracy
    print("History of (hyperparameter,meanAccuracy(innerCrossVal.)): \n", all_accuracies)
    return best_h
     
def doDoubleCrossValidation (D, k, H):
    allIdxs = np.arange(len(D))
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    print("Outer K folds:\n" , idxs)
    accuracies = []
    for o_fold in range(k):
        print(f"\n\n---------- OUTER FOLD {o_fold+1} ----------\n")
        # Get all indexes for this fold
        testIdxs = idxs[o_fold,:]
        # Get all the other indexes
        trainIdxs = np.array(list(set(allIdxs) - set(testIdxs))).flatten()
        best_h = get_best_hyperparameters(D[trainIdxs],k,H)
        print("Best hyper parameter: ",best_h)
        # Train the model on the training data
        model = trainModel(D[trainIdxs], best_h)
        # Test the model on the testing data
        accuracies.append(testModel(model, D[testIdxs]))
    print("\n------------------------FINAL RESULT------------------------\n")
    print(f"History of Accuracies for Outer Folds with best h :", accuracies,"\nAverage: ", np.mean(accuracies),"\n")

    return np.mean(accuracies)


if __name__=='__main__':
    main()