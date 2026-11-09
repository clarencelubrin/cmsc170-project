"""
References: https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/,
            https://www.youtube.com/watch?v=w8yWXqWQYmU - Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math) by Samson Zhang
"""
import math
from random import gauss
from numpy import transpose

def init_params(inputNum, hiddenNum):
    """
    Initialize the starting weights for a 1-hidden-layer perceptron.

    input/hidden_std: the std used for initialization.
    w1              : a 2d list of weights for the connection between the input neurons and the hidden neurons.
    b1              : a list of biases for all the connections from input to hidden.
    w2              : a list of weights for the connection between the hidden neurons and the output neuron.
    b2              : a value for the bias between the connection of hidden neurons and the output neuron

    Reference: https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/, https://www.geeksforgeeks.org/python/random-gauss-function-in-python/, https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/,
    """
    
    input_std = math.sqrt(2 / inputNum)
    w1 = [[gauss(0.0, input_std) for _ in range(inputNum)] for _ in range(hiddenNum)]
    b1 = [0.0] * hiddenNum

    hidden_std = math.sqrt(2  / hiddenNum)
    w2 = [gauss(0.0, hidden_std) for _ in range(hiddenNum)]  
    b2 = 0.0

    return w1, b1, w2, b2

def forward_propagation(w1, b1, w2, b2, X):
    """
    Implementation of the forward propagation.

    z1: weighted sum plus bias of the first layer (input -> hidden)
    a1: pass z1 into the activation function (RelU)
    z2: weighed sum plus bias of the second layer (hidden -> output)
    a2: pass z2 into the activation function (sigmoid)

    Reference: https://www.datacamp.com/tutorial/forward-propagation-neural-networks
    """
    hiddenNum = len(w1)
    Z1, A1, Z2, A2 = [], [], [], []
    for x in X:
        z1 = [dot(w1[h], x) + b1[h] for h in range(hiddenNum)]
        a1 = [RelU(z) for z in z1]
        z2 = dot(w2, a1) + b2
        a2 = softmax(z2)
        Z1.append(z1) 
        A1.append(a1)
        Z2.append(z2)
        A2.append(a2)
    return Z1, A1, Z2, A2

def back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y): # Need to implement labels, KEEP IN MIND THAT WE MUST TRANSPOSE THE MATRIX
    dZ2 = A2 - Y                                    # SO Y IS ACTUALLY THE 1st OR LAST ROW INSTEAD OF IT BEING THE COLUMN.
    dW2 = 1 / len(X) * dot(dZ2, A1.T)
    dB2 = 1 / len(X) * sum(dZ2)
    return

def one_hot(Y): # Transform the labels from [1,2,3] -> [0,1,0] (basically 1 if it pertains to this label, 0 if not.)
    one_hot_Y = [[0 for _ in range(len(Y))] for _ in range(max(Y) + 1)]
    
    return
def dot(w, x): # dot product
    total = 0.0
    for i in range(len(w)):
        total += w[i] * x[i]
    return total

"""
Reference: https://www.geeksforgeeks.org/deep-learning/choosing-the-right-activation-function-for-your-neural-network/
"""
# def sigmoid(z): # For output layer, since current implementation of classification is binary
#     return 1/(1+math.exp(-z))

def softmax(z): # For the output layer, activation function for multiple classifications.
    for output in z:
        break
    return 

def RelU(z): # For hidden layer, simple implementation (switch to other activation function if needed) P.S. look into using tanh instead since it's better for the z-scored inputs.
    return max(0,z)

def deriv_RelU(z):
    return int(z > 0) # if z is greater than 0, return 1. Else, return 0. 

