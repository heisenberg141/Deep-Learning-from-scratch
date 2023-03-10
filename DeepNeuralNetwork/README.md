# Description
This Directory Contains implementation of arbitrarily deep neural network, training and testing for a multi class classification task with cross entropy loss and Stochastic Gradient Descent.

Download testing and training data from the following links:
1. X_train: [https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy)
2. Y_train:[https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy)
3. X_test: [https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy)
4. Y_test: [https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy)

## Running the code
To run the code:
1. Download the dataset mentioned above into this directory
2. Install python3.5+, matplotlib, numpy and scipy.
3. Run the code python3 softmax_regression.py.

## Results
The code produces an accuracy of 88.09%. Below is the visualization of the template learned in the first layer of the network.


![Results](Result.png)



