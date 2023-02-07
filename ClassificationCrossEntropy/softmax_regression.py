import numpy as np
import matplotlib.pyplot as plt

def main():
    softmax_runner()

def softmax_runner():
    X_train, X_test, Y_train, Y_test = load_data()
    batch_sizes =   [512,1024,2048,8192]
    learning_rates= [0.001, 0.01 ,0.1]
    epochs = [10,20]
    alphas = [0.001, 0.01, 0.1]
    H = list()
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            for k in range(len(epochs)):
                for l in range(len(alphas)):
                    H.append([int(batch_sizes[i]),learning_rates[j],int(epochs[k]),alphas[l]])
    print(f"Total H parameters: {len(H)}")
    # star_h = find_best_hps(X_train, Y_train, H) 
    star_h =[1024, 0.1,20,0.05]
    model = train_model(X_train, Y_train,star_h)
    w,b = model
    test_model(model,X_test, Y_test)
    visualize_model(model)

def find_best_hps(X_tr,y_tr, H):
    allidxs = np.arange(X_tr.shape[1])
    allidxs = np.random.permutation(allidxs)
    X_tr = X_tr[:,allidxs]
    y_tr = y_tr[:,allidxs]

    val_split = 0.2
    X_train, y_train, X_val, y_val = train_val_split(X_tr,y_tr,val_split)
    print(f"\nX train: {X_train.shape}")
    print(f"Y train: {y_train.shape}")
    print(f"X Val: {X_val.shape}")
    print(f"Y Val: {y_val.shape}")
    star_h = [0,0,0,0]
    best_loss = 1e6

    c = y_tr.shape[0]
    m = X_tr.shape[0]
    star_h = [0,0,0,0]
    best_loss = 1e6
    for h in H:
        # batch size, learning rate, epochs, alpha
        batch_size, lr, epochs, alpha = h
        w = np.random.normal(0, 0.1, (c,m))
        b = np.random.normal(0,0.1,(c,1))

        print(f"Training with h = {h}. ")
        
        for epoch in range(epochs):
            mini_batches = generate_mini_batches(X_train, y_train, batch_size)
            for batch in mini_batches:
                x,y = batch
                n = x.shape[1]
                
                z = np.dot(w, x) + b.repeat(n,axis = 1)
                temp = np.exp(z)
                y_hat = temp/np.sum(temp, axis = 0)

                w_gradient = 1/n* np.dot((y_hat-y),x.T) + alpha*w
                b_gradient = np.mean(y_hat-y, axis = 1)

                w -= lr * w_gradient
                b -= lr * b_gradient.reshape((c,1))
        loss = test_model((w,b),X_val, y_val)
        if loss<best_loss:
            best_loss = loss
            star_h = h
        print("__________________________________________________________________\n")
        
    print("\n____________________________DONE___________________________________\n")
    print(f"BEST CE loss: {best_loss}\nBest Hyper Parameters: {star_h}\n\t(batch size, learning rate, epochs, alpha)\n")
    return star_h
        
def generate_mini_batches(X, Y, batch_size):
    m = X.shape[0]
    c = Y.shape[0]
    n = X.shape[1]
    full_batches = n//batch_size
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

def train_val_split(X_tr,y_tr,val_split): 
    n = X_tr.shape[1]
    print(val_split*len(X_tr.T))
    X_val = X_tr.T[:int(val_split*n),].T
    y_val = y_tr.T[:int(val_split*n),].T
    X_train = X_tr.T[int(val_split*n):,].T
    y_train = y_tr.T[int(val_split*n):,].T
    return X_train, y_train, X_val, y_val

def train_model(X, Y, h):
    '''
    X: m x n
    Y: c x n
    b: c x 1
    W(to be trained): c x m
    Z = W.X + b(repeated n times column wise)
    Z: c x n
    '''
    allidxs = np.arange(X.shape[1])
    allidxs = np.random.permutation(allidxs)
    X = X[:,allidxs]
    Y = Y[:,allidxs]

    c = Y.shape[0]
    m = X.shape[0]
    batch_size, lr, epochs, alpha = h
    
    w = np.random.normal(0, 0.1, (c,m))
    b = np.random.normal(0,0.1,(c,1))

    print(f"Training with h = {h}. ")
    
    for epoch in range(epochs):
        # print(f"Training epoch: {epoch+1}")
        mini_batches = generate_mini_batches(X, Y, batch_size)
        for batch in mini_batches:
            x,y = batch
            n = x.shape[1]
            
            z = np.dot(w, x) + b.repeat(n,axis = 1)
            temp = np.exp(z)
            y_hat = temp/np.sum(temp, axis = 0)

            w_gradient = 1/n* np.dot((y_hat-y),x.T) + alpha*w
            b_gradient = np.mean(y_hat-y, axis = 1)

            w -= lr * w_gradient
            b -= lr * b_gradient.reshape((c,1))
    
    return w, b

def test_model(model,X,Y):
    print("\nTesting:")
    W,b = model
    Z = np.dot(W, X) + b 
    temp = np.exp(Z)
    Y_hat = temp/np.sum(temp, axis = 0)

    log_y_hat = np.log(Y_hat)
    CE = -np.sum(Y*log_y_hat)/Y.shape[0]
    print(f"\tCE: {CE}")
    correct = 0
    total = 0
    for i in range(len(Y.T)):
        if np.argmax(Y.T[i])==np.argmax(Y_hat.T[i]):
            correct+=1
        total+=1
    acc = correct/total
    print("\tAccuracy: " , acc)
    return CE
    

def load_data():
    X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 784))
    X_tr_normalized = X_tr/255
    y_tr = np.load("fashion_mnist_train_labels.npy")

    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 784))
    X_te_normalized = X_te/255
    y_te = np.load("fashion_mnist_test_labels.npy")
    
    Y_tr_one_hot = np.zeros((y_tr.shape[0],10))
    Y_tr_one_hot[np.arange(Y_tr_one_hot.shape[0]), y_tr] = 1
    
    Y_te_one_hot = np.zeros((y_te.shape[0],10))
    Y_te_one_hot[np.arange(Y_te_one_hot.shape[0]), y_te] = 1
    

    # print(f"X_tr: {X_tr.shape}")
    # print(f"Y_tr: {y_tr.shape}")
    # print(f"X_te: {X_te.shape}")
    # print(f"Y_te: {y_te.shape}")

    return X_tr_normalized.T, X_te_normalized.T, Y_tr_one_hot.T, Y_te_one_hot.T

def visualize_X(X):
    print("Testing")
    assert np.max(X)<=1
    assert np.min(X)>=0
    print("Dataset_max_value: ",np.max(X))
    print("Dataset_min_value: ",np.min(X))
    
    idx = np.random.randint(0,X.shape[0],16)
    nrows = 4
    ncols = 4
    fig, ax = plt.subplots(nrows,ncols,figsize = (5,5))
    
    ax = ax.flatten()
    for i in range(nrows*ncols):
        pic = X[idx[i]].reshape(28,28)
        ax[i].imshow(pic,cmap='gray')
        ax[i].set_axis_off()
    plt.show()

def visualize_model(model):
    '''
    W: c x m
    
    '''
    W,b = model
    
   
    nrows = 2
    ncols = 5

    fig, ax = plt.subplots(nrows,ncols,figsize = (5,5))
    ax = ax.flatten()
    i = 0
    for template in W:
        template+=b[i]
        template = template.reshape([28,28])
        ax[i].imshow(template,cmap='gray')
        ax[i].set_axis_off()
        i+=1
    plt.show()

    

if __name__ == '__main__':
    main()