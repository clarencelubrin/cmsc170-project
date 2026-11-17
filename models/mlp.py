"""
References: https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/,
            https://www.youtube.com/watch?v=w8yWXqWQYmU - Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math) by Samson Zhang
"""
import math
from random import gauss

def transpose(M):
    return [list(col) for col in zip(*M)]

def init_params(inputNum, hiddenNum, outputNum):
    """
    Initialize the starting weights for a 1-hidden-layer perceptron.

    input/hidden_std: the std used for initialization.
    w1              : a 2d list of weights for the connection between the input neurons and the hidden neurons.
    b1              : a list of biases for all the connections from input to hidden.
    w2              : a list of weights for the connection between the hidden neurons and the output neuron.
    b2              : a list of biases for all the connections of hidden neurons and the output neuron

    Reference: https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/, https://www.geeksforgeeks.org/python/random-gauss-function-in-python/, https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/,
    """
    
    input_std = math.sqrt(2 / inputNum)
    w1 = [[gauss(0.0, input_std) for _ in range(inputNum)] for _ in range(hiddenNum)]
    b1 = [0.0] * hiddenNum

    hidden_std = math.sqrt(2  / hiddenNum)
    w2 = [[gauss(0.0, hidden_std) for _ in range(hiddenNum)] for _ in range(outputNum)]  
    b2 = [0.0] * outputNum

    return w1, b1, w2, b2

def forward_propagation(w1, b1, w2, b2, X):
    """
    Implementation of the forward propagation.

    z1: weighted sum plus bias of the first layer (input -> hidden)
    a1: pass z1 into the activation function (RelU)
    z2: weighed sum plus bias of the second layer (hidden -> output)
    a2: pass z2 into the activation function (Softmax)

    Reference: https://www.datacamp.com/tutorial/forward-propagation-neural-networks
    """
    hiddenNum = len(w1)
    outputNum = len(w2)
    Z1, A1, Z2, A2 = [], [], [], []
    for x in X:
        z1 = [dot(w1[h], x) + b1[h] for h in range(hiddenNum)]
        a1 = [Tanh(z) for z in z1]
        z2 = [dot(w2[o], a1) + b2[o] for o in range(outputNum)]
        a2 = softmax(z2)
        Z1.append(z1) 
        A1.append(a1)
        Z2.append(z2)
        A2.append(a2)
    return Z1, A1, Z2, A2

def one_hot(Y, num_classes): # Transform the label 1 -> [0,1,0] in a 3 class (0, 1, 2) example (basically 1 if it pertains to this label, 0 if not.)
    one_hot_Y = [[0] * num_classes for _ in range(len(Y))]
    for i in range(len(Y)):
        one_hot_Y[i][Y[i]] = 1
    return one_hot_Y

# From Samson Zhang's MNIST NN from Scratch
def back_propagation(Z1, A1, Z2, A2, w1, w2, X, Y):
    m = len(X)
    hiddenNum, outputNum, inputNum = len(w1), len(w2), len(X[0])
    X_T, A1_T = transpose(X), transpose(A1)

    one_hot_Y = one_hot(Y, outputNum)
    dZ2 = [[A2[s][o] - one_hot_Y[s][o] for o in range(outputNum)] for s in range(m)] 
    dZ2_T = transpose(dZ2)
    dW2 = [[1 / m * dot(dZ2_T[o], A1_T[h]) for h in range(hiddenNum)] for o in range(outputNum)]
    db2 = [1 / m * sum(row) for row in dZ2_T]

    w2_T = transpose(w2)
    dZ1 = [[dot(w2_T[h], dZ2[s]) * deriv_Tanh(Z1[s][h]) for h in range(hiddenNum)] for s in range(m)]
    dZ1_T = transpose(dZ1)
    dW1 = [[1 / m * dot(dZ1_T[h], X_T[i]) for i in range(inputNum)] for h in range(hiddenNum)]
    db1 = [1 / m * sum(row) for row in dZ1_T]
    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 = [[w - alpha * d for w, d in zip(wr, dr)] for wr, dr in zip(w1, dW1)]
    b1 = [b - alpha * d for b, d in zip(b1, db1)]
    w2 = [[w - alpha * d for w, d in zip(wr, dr)] for wr, dr in zip(w2, dW2)]
    b2 = [b - alpha * d for b, d in zip(b2, db2)]
    return w1, b1, w2, b2

def get_predictions(A2):
    return [max(range(len(a)), key=lambda c: a[c]) for a in A2]

def get_accuracy(predictions, Y):
    return sum(p == y for p, y in zip(predictions, Y)) / len(Y)

def make_predictions(X, w1, b1, w2, b2):
    _, _, _, A2 = forward_propagation(w1, b1, w2, b2, X)
    return get_predictions(A2)
    
def dot(w, x): # dot product
    total = 0.0
    for i in range(len(w)):
        total += w[i] * x[i]
    return total

def gradient_descent(X, Y, alpha, iterations, hiddenNum, outputNum):
    w1, b1, w2, b2 = init_params(len(X[0]), hiddenNum, outputNum)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 2 == 0:
            print("Iteration: ", i)
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y) * 100}%")
    return w1, b1, w2, b2

"""
ACTIVATION FUNCTIONS SECTION

Reference: https://www.geeksforgeeks.org/deep-learning/choosing-the-right-activation-function-for-your-neural-network/, https://www.geeksforgeeks.org/deep-learning/tanh-activation-in-neural-network/
"""
# def sigmoid(z): # For output layer, since current implementation of classification is binary
#     return 1/(1+math.exp(-z))

def softmax(z): # For the output layer, activation function for multiple classifications.
    m = max(z)                          
    exps = [math.exp(v) for v in z]
    total = sum(exps)
    return [e / total for e in exps]

def RelU(z): # For hidden layer, simple implementation (switch to other activation function if needed) P.S. look into using tanh instead since it's better for the z-scored inputs.
    return max(0,z)

def deriv_RelU(z):
    return int(z > 0) # if z is greater than 0, return 1. Else, return 0. 

def Tanh(z):
    return math.tanh(z)

def deriv_Tanh(z):
    return 1 - math.tanh(z) ** 2