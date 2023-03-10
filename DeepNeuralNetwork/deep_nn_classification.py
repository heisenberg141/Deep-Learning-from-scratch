import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 64 ]
NUM_OUTPUT = 10
REGULARIZER = 0.0

def main():
    # Load the training data.
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).
    trainX, testX, trainY, testY = load_data()
    
    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    Ws, bs = initWeightsAndBiases()
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    
    # gradient_check(trainX, trainY,weights)

    list_params = get_hyper_parameters()
    # h_star = find_best_hps(trainX, trainY,list_params)
    # batch size, lr, epochs, alpha
    h_star = [32, 0.01, 30, 0.001] #Best for 3 hidden layers with 64 units each
    print(f"H: {h_star}\n")
    weights,ce_loss,accuracy = train(trainX, trainY, weights, testX, testY, h_star)
    
    show_W0(weights)

def find_best_hps(X_tr,y_tr, H):
    allidxs = np.arange(X_tr.shape[1])
    allidxs = np.random.permutation(allidxs)
    X_tr = X_tr[:,allidxs]
    y_tr = y_tr[:,allidxs]

    val_split = 0.2
    trainX, trainY, testX, testY = train_val_split(X_tr,y_tr,val_split)
   
    star_h = [0,0,0,0]
    best_loss = 1e6
    print()
    i = 0
    for h in H:
        print(f"Experiment No. {i+1}")
        i+=1
        print(f"Training with h = {h}. ")
        Ws, bs = initWeightsAndBiases()
        weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
        _,loss,accuracy = train(trainX, trainY, weights, testX, testY, h)
        print(f"Done training.\n1. Loss: {loss}\n2. Accuracy: {accuracy*100}%\n")
        
        if loss<best_loss:
            best_loss = loss
            star_h = h
        print("_______________________________________________________\n")
        
    print("\n____________________________DONE___________________________________\n")
    print(f"BEST CE loss: {best_loss}\nBest Hyper Parameters: {star_h}\n\t(batch size, learning rate, epochs, alpha)\n")
    return star_h

def gradient_check(trainX, trainY, weights):
    # On just the first n training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).
    n = 5
    print(f"Checking error over {n} training examples\n")
    print("Gradient Error: ",scipy.optimize.check_grad(lambda weights_: fCE(np.atleast_2d(trainX[:,0:n]), np.atleast_2d(trainY[:,0:n]), weights_), \
                                    lambda weights_: gradCE_for_one(np.atleast_2d(trainX[:,0:n]), np.atleast_2d(trainY[:,0:n]), weights_), \
                                    weights))
    print("\n========================================================\n")
    input("\nPress enter to start training:\n")

    # approx_gradients = scipy.optimize.approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[:,0:n]), np.atleast_2d(trainY[:,0:n]), weights_), 1e-6)

def train_val_split(X_tr,y_tr,val_split): 
    n = X_tr.shape[1]
    X_val = X_tr.T[:int(val_split*n),].T
    y_val = y_tr.T[:int(val_split*n),].T
    X_train = X_tr.T[int(val_split*n):,].T
    y_train = y_tr.T[int(val_split*n):,].T
    return X_train, y_train, X_val, y_val

def get_hyper_parameters():
    batch_sizes =   [64,128, 256, 512]
    learning_rates= [0.005, 0.05, 0.5 ]
    epochs = [20,30]
    alphas = [0.001, 0.01, 0.1]
    H = list()
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            for k in range(len(epochs)):
                for l in range(len(alphas)):
                    H.append([int(batch_sizes[i]),learning_rates[j],int(epochs[k]),alphas[l]])
    # print(f"Total H parameters: {len(H)}")
    return H

def train (trainX, trainY, weights, testX, testY, h_star):
    allidxs = np.arange(trainX.shape[1])
    allidxs = np.random.permutation(allidxs)
    trainX = trainX[:,allidxs]
    trainY = trainY[:,allidxs]

    batch_size, lr, epochs, alpha = h_star
    i = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        mini_batches = generate_mini_batches(trainX, trainY, batch_size)
        
        for batch in mini_batches:
            # print("\tupdating weights")
            x,y = batch
            weights -= lr * gradCE(x,y,weights,alpha)
            i+=1
        
        Ws,bs = unpack(weights)
        y_hat, _ = forward(Ws, bs, testX)
        log_y_hat = np.log(y_hat)
        ce = -np.sum(testY*log_y_hat)/testY.shape[1]
        print(f"CE loss on test data: {ce}")
        correct = 0
        total = 0
        for i in range(len(testY.T)):
            if np.argmax(testY.T[i])==np.argmax(y_hat.T[i]):
                correct+=1
            total+=1
        acc = correct/total
        print(f"Test Accuracy after {epoch+1} epoch: {acc*100}%\n")

    print(f"Total updates: {i + 1} ")

    Ws,bs = unpack(weights)
    y_hat, _ = forward(Ws, bs, testX)
    log_y_hat = np.log(y_hat)
    ce = -np.sum(testY*log_y_hat)/testY.shape[1]
    
    correct = 0
    total = 0
    for i in range(len(testY.T)):
        if np.argmax(testY.T[i])==np.argmax(y_hat.T[i]):
            correct+=1
        total+=1
    acc = correct/total
    print(f"Accuracy: {acc*100}%")
    

        # break
    return weights,ce,acc

def generate_mini_batches(X, Y, batch_size):
    batch_size = int(batch_size)
    m = X.shape[0]
    c = Y.shape[0]
    n = X.shape[1]
    full_batches = int(n//batch_size)
    mini_batches = list()
    for ii in range(full_batches):
        X_mini_batch = X[:,ii*batch_size:(ii+1)*batch_size]
        Y_mini_batch = Y[:,ii*batch_size:(ii+1)*batch_size]
        mini_batches.append((X_mini_batch,Y_mini_batch))

    if n % batch_size != 0:
        X_mini = X[:,batch_size*full_batches:]
        Y_mini = Y[:,batch_size*full_batches:]
        mini_batches.append((X_mini,Y_mini))
    
    # for batch in mini_batches:
    #     print(f"X_batch: {batch[0].shape}\nY_batch: {batch[1].shape}\n")

    # X_mini_batches = X[:,:full_batches*batch_size].reshape((-1,m,batch_size))
    # Y_mini_batches = Y[:,:full_batches*batch_size].reshape((-1,c,batch_size))
    return mini_batches

def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

def fCE (X, Y, weights):
    Ws, bs = unpack(weights)
    y_hat, _ = forward(Ws, bs, X)
    log_y_hat = np.log(y_hat)
    ce = -np.sum(Y*log_y_hat)/Y.shape[1]
    return ce

def forward(Ws, bs, X):
    Z_list = list()
    Z_list.append(X)
    H_list = list()
    H_list.append(X)
    
    layer_output = X
    for i in range(NUM_HIDDEN_LAYERS):
        '''
        X: m x n
        W = c x m
        b = c x 1
        '''
        
        m = layer_output.shape[0]
        n = layer_output.shape[1]
        c = Ws[i].shape[0]

        # print(f"W{i+1}: {Ws[i].shape}")
        # print(f"b{i+1}: {bs[i].shape}")
        
        layer_output = np.dot(Ws[i],layer_output) + bs[i].reshape(c,1).repeat(n,axis = 1)
        Z_list.append(layer_output)
        # print("Before ReLu: ",layer_output.shape)
        layer_output = np.maximum(layer_output, 0)
        H_list.append(layer_output)
        # print("After ReLu: ",layer_output.shape)
    
    c = NUM_OUTPUT
    last_Z= np.dot(Ws[-1],layer_output) + bs[-1].reshape(c,1).repeat(n,axis = 1)
    # print("last_layer: ", last_Z.shape)
    Z_list.append(last_Z)
    temp = np.exp(last_Z)
    y_hat = temp/np.sum(temp, axis = 0)
    H_list.append(y_hat)
    # print("y_hat: ",y_hat.shape)
    # print("sum(y_hat)", np.sum(y_hat,axis=0))
    backprop_lists = [Z_list, H_list]
    return y_hat, backprop_lists

def gradCE_for_one(X,Y, weights):

    '''
    NETWORK STRUCTURE
    H_LIST[0]/Z_LIST[0] --Ws[0],bs[0]--> Z_LIST[1]--relu-->H_LIST[1] ... --Ws[3],bs[3]--> W_list[4]-CE->H_list[4] 
    
    '''
    Ws,bs = unpack(weights)
    y_hat, backprop_lists = forward(Ws, bs, X)
    Z_list, H_list = backprop_lists
    # print("weights: ",len(backprop_lists[0]))
    g = None
    Grad_Ws = list()
    Grad_bs = list()
    for i in range(NUM_HIDDEN_LAYERS+1, 0, -1):
        print(f"BackProp Iter {i}")
        if i == NUM_HIDDEN_LAYERS +1:
            print("1. INIT STEP: Since it is the first iteration, we will update g as (y_hat- y)")
            g = y_hat-Y
            
        else:
            
            print(f"1. We will perform hadamard product of g with relu\'(Z{i})")
            Zi = Z_list[i]
            relu_prime_Zi = (Zi>0).astype(Zi.dtype)
            g = g * relu_prime_Zi
        
        # print(f"H_list[{i-1}]: {H_list[i-1].shape}")
        print(f"2. Grad_W{i} = g . h{i-1}.T + alpha * grad_W{i}(regularizer)")
        gradient_W_i = (np.dot(g, H_list[i-1].T) + REGULARIZER* Ws[i-1])/g.shape[1] 
        Grad_Ws.append(gradient_W_i)
        print(f"3. Grad_b{i} = g + alpha * grad_b{i}(regularizer)")
        gradient_b_i = np.sum(g, axis = 1)/g.shape[1]
        Grad_bs.append(gradient_b_i)

        print(f"4. Next, g will be pre multiplied by W{i}.T ")
        g = np.dot(Ws[i-1].T,g)
        
        print("________________________________________________________\n")

    # print(f"Final G: {g.shape}")
    Grad_Ws.reverse()
    Grad_bs.reverse()
    gradient_weights = np.hstack([ grad_w.flatten() for grad_w in Grad_Ws] + [ grad_b.flatten() for grad_b in Grad_bs])
    return gradient_weights
        
    
    # return gradients

def gradCE (X, Y, weights, alpha):

    Ws,bs = unpack(weights)
    y_hat, backprop_lists = forward(Ws, bs, X)
    Z_list, H_list = backprop_lists
    g = None
    Grad_Ws = list()
    Grad_bs = list()
    for i in range(NUM_HIDDEN_LAYERS+1, 0, -1):
        if i == NUM_HIDDEN_LAYERS +1:
            g = y_hat-Y
        else:
            Zi = Z_list[i]
            relu_prime_Zi = (Zi>0).astype(Zi.dtype)
            g = g * relu_prime_Zi
        
        gradient_W_i = (np.dot(g, H_list[i-1].T) + alpha * Ws[i-1]) /g.shape[1] 
        Grad_Ws.append(gradient_W_i)

        gradient_b_i = np.sum(g, axis = 1)/g.shape[1]
        Grad_bs.append(gradient_b_i)

        g = np.dot(Ws[i-1].T,g)
        

    Grad_Ws.reverse()
    Grad_bs.reverse()
    gradient_weights = np.hstack([ grad_w.flatten() for grad_w in Grad_Ws] + [ grad_b.flatten() for grad_b in Grad_bs])
    return gradient_weights
   
# Creates an image representing the first layer of weights (W0).
def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()

def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i], NUM_HIDDEN[i+1]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs

def load_data():
    X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 784))
    X_tr_normalized = X_tr/255 -0.5
    y_tr = np.load("fashion_mnist_train_labels.npy")

    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 784))
    X_te_normalized = X_te/255 -0.5
    y_te = np.load("fashion_mnist_test_labels.npy")
    
    Y_tr_one_hot = np.zeros((y_tr.shape[0],10))
    Y_tr_one_hot[np.arange(Y_tr_one_hot.shape[0]), y_tr] = 1
    
    Y_te_one_hot = np.zeros((y_te.shape[0],10))
    Y_te_one_hot[np.arange(Y_te_one_hot.shape[0]), y_te] = 1
    
    # print(f"MaxVal: {max(X_tr_normalized[0])}\nMinVal: {min(X_tr_normalized[0])}")
    # print(f"X_tr: {X_tr.shape}")
    # print(f"Y_tr: {y_tr.shape}")
    # print(f"X_te: {X_te.shape}")
    # print(f"Y_te: {y_te.shape}")

    return X_tr_normalized.T, X_te_normalized.T, Y_tr_one_hot.T, Y_te_one_hot.T


if __name__ == "__main__":
    main()
        

