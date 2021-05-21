import numpy as np
import matplotlib.pyplot as plt

"""
q1a.py

This program implements a single hidden layer MLP 
and performs gradient-checking

@author: Anushree Sitaram Das (ad1707)
"""


def softmax(x):
    """
    Activation function for regression model.
    It takes as input a vector z of K real numbers,
    and normalizes it into a probability distribution consisting of
    K probabilities proportional to the exponentials of the input numbers.
    :param X:   input array
    :return:
    """
    return np.exp(x) / np.sum(np.exp(x),axis=1, keepdims=True)


def sigmoid(x):
    """
        Returns value between 0 and 1
        :param x:
        :return:
    """
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    """
        Returns derivative of sigmoid(x)
        :param x:
        :return:
    """
    return sigmoid(x) *(1-sigmoid (x))


def loss_function(W1, b1, W2, b2, X, y,lamb=1):
    """
    Calculates loss using cost function and
    calculates the cost function's partial derivative
    to get parameter's gradient update values

    :param W1:      Input Weights for Hidden Layer
    :param b1:      Input Bias for Hidden Layer
    :param W2:      Input Weights for Output Layer
    :param b2:      Input Bias for Output Layer
    :param X:       Input Features
    :param y:       Output Class
    :param lamb:    lambda value
    :return: Loss, partial gradient descents for all parameters
    """

    N = X.shape[0]

    # feedforward
    # (input features . weights 1) + bias 1
    z1 = np.dot(X, W1) + b1
    # activation at hidden layer
    a1 = sigmoid(z1)
    # (output from hidden layer . weights 2) + bias 2
    z2 = np.dot(a1, W2) + b2
    # activation at output layer
    a2 = softmax(z2)

    # cross-entropy loss without regualrization
    loss = - (np.sum(y * np.log(a2)) / N)

    # backpropagation

    a2_delta = (a2 - y) / N  # w2
    z1_delta = np.dot(a2_delta, W2.T)
    a1_delta = z1_delta * sigmoid_derivative(a1)  # w1

    # gradients for all parameters
    gradW2 = (1/N)*(np.dot(a1.T, a2_delta)) + (lamb * W2)
    gradb2 = np.sum(a2_delta, axis=0, keepdims=True)
    gradW1 = (1/N)*(np.dot(X.T, a1_delta)) + (lamb * W1)
    gradb1 = np.sum(a1_delta, axis=0)

    return loss, gradW1, gradb1, gradW2, gradb2


def vertor_to_matrix(y):
    """
    Converts output class vector to matrix.
    Ex:
    vector = [0,1,2,1]
    matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]]
    :param y:   vector
    :return:    matrix
    """
    return (np.arange(np.max(y) + 1) == y[:, None]).astype(float)



def predict(X, W1, b1, W2, b2):
    """
    Predict class for given vector of features

    :param X: Input features
    :param W1:      Input Weights for Hidden Layer
    :param b1:      Input Bias for Hidden Layer
    :param W2:      Input Weights for Output Layer
    :param b2:      Input Bias for Output Layer
    :return:  Labels for each given input features
    """
    # feedforward
    # (input features . weights 1) + bias 1
    z1 = np.dot(X, W1) + b1
    # activation at hidden layer
    a1 = sigmoid(z1)
    # (output from hidden layer . weights 2) + bias 2
    z2 = np.dot(a1, W2) + b2
    # activation at output layer
    a2 = softmax(z2)
    # return index of the element with maximum value
    return a2.argmax(axis=1)


def get_accuracy(X,y,W1,b1,W2,b2):
    """
    Calculates accuracy of predictions for the given model
    :param X: Input features
    :param y: Output class
    :param W1:      Input Weights for Hidden Layer
    :param b1:      Input Bias for Hidden Layer
    :param W2:      Input Weights for Output Layer
    :param b2:      Input Bias for Output Layer
    :return:  Accuracy in percentage
    """
    pred = predict(X,W1,b1,W2,b2)
    sum = 0
    for i in range(len(pred)):
        if pred[i]==y[i]:
            sum+=1
    accuracy = sum/(float(len(y)))
    return accuracy*100


def get_parameters(X,Y,epochs,learningRate):
    """
    Calculates optimal values for weights and bias via regression

    :param X:               Input features
    :param Y:               Output class
    :param epochs:          Number of epochs
    :param learningRate:    Learning rate for this model
    :return:                Optimal weights and bias
    """
    num_labels = len(np.unique(Y))
    # number of units in the hidden layer
    num_units = 20
    d = np.sqrt(1 / (X.shape[1] + num_units))
    # initialize Weights 1
    W1 = np.random.randn(X.shape[1], num_units) * d
    # initialize Bias 1
    b1 = np.zeros(num_units)

    # initialize Weights 2
    W2 = np.random.randn(num_units, num_labels) * d
    # initialize Bias 2
    b2 = np.zeros(num_labels)

    # convert output vector to matrix
    y = vertor_to_matrix(Y)

    # stores loss for each epoch
    losses = []

    for i in range(0, epochs):
        # get loss and gradient descents for weights and bias
        loss, gradW1, gradb1, gradW2, gradb2 = loss_function(W1,b1,W2,b2, X, y)
        losses.append(loss)
        # update weights and bias
        W1 = W1 - (learningRate * gradW1)
        b1 = b1 - (learningRate * gradb1)
        W2 = W2 - (learningRate * gradW2)
        b2 = b2 - (learningRate * gradb2)

    # Plot Losses
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss over epochs")
    plt.savefig('losses_q1a.png')
    plt.show()

    return W1,b1,W2,b2



def gradient_checking(X, Y, W1, b1, W2, b2):
    """
    Performs gradient-checking.

    :param X: Input features
    :param y: Output class
    :param W1:      Input Weights for Hidden Layer
    :param b1:      Input Bias for Hidden Layer
    :param W2:      Input Weights for Output Layer
    :param b2:      Input Bias for Output Layer
    :return:  None
    """
    # convert output vector to matrix
    y = vertor_to_matrix(Y)
    epsilon = 10e-4

    # check approximations of derivatives for  weights 1
    print("Checking gradients of Weights 1:")
    for i in range(len(W1)):
        for j in range(len(W1[0])):
            Wnew = W1.copy()

            Wnew[i][j] = W1[i][j] + epsilon
            Jplus, gradW1, gradb1, gradW2, gradb2 = loss_function(Wnew, b1, W2, b2, X, y)

            Wnew[i][j] =  W1[i][j] - epsilon
            Jminus, gradW1, gradb1, gradW2, gradb2 = loss_function(Wnew, b1, W2, b2, X, y)

            # approximation of derivative
            if (Jplus - Jminus) / (2 * epsilon) < 1e-3:
                print("CORRECT")
            else:
                print("INCORRECT", (Jplus - Jminus) / (2 * epsilon))

    # check approximations of derivatives for bias 1
    print("Checking gradients of Bias 1:")
    for i in range(len(b1)):
        bnew = b1.copy()

        bnew[i] = b1[i] + epsilon
        Jplus, gradW1, gradb1, gradW2, gradb2 = loss_function(W1, bnew, W2, b2, X, y)

        bnew[i] =  b1[i] - epsilon
        Jminus, gradW1, gradb1, gradW2, gradb2 = loss_function(W1, bnew, W2, b2, X, y)

        # approximation of derivative
        if (Jplus - Jminus) / (2 * epsilon) < 1e-3:
            print("CORRECT")
        else:
            print("INCORRECT", (Jplus - Jminus) / (2 * epsilon))

    # check approximations of derivatives for weights 2
    print("Checking gradients of Weights 2:")
    for i in range(len(W2)):
        for j in range(len(W2[0])):
            Wnew = W2.copy()

            Wnew[i][j] = W2[i][j] + epsilon
            Jplus, gradW1, gradb1, gradW2, gradb2 = loss_function(W1, b1, Wnew, b2, X, y)

            Wnew[i][j] = W2[i][j] - epsilon
            Jminus, gradW1, gradb1, gradW2, gradb2 = loss_function(W1, b1, Wnew, b2, X, y)

            # approximation of derivative
            if (Jplus - Jminus) / (2 * epsilon) < 1e-3:
                print("CORRECT")
            else:
                print("INCORRECT", (Jplus - Jminus) / (2 * epsilon))

    # check approximations of derivatives for bias 2
    print("Checking gradients of Bias 2:")
    for i in range(len(b2)):
        bnew = b2.copy()

        bnew[i] = b2[i] + epsilon
        Jplus, gradW1, gradb1, gradW2, gradb2 = loss_function(W1, b1, W2, bnew, X, y)

        bnew[i] = b2[i] - epsilon
        Jminus, gradW1, gradb1, gradW2, gradb2 = loss_function(W1, b1, W2, bnew, X, y)

        # approximation of derivative
        if (Jplus - Jminus) / (2 * epsilon) < 1e-3:
            print("CORRECT")
        else:
            print("INCORRECT", (Jplus - Jminus) / (2 * epsilon))


if __name__ == "__main__":
    # load data
    data = np.genfromtxt('xor.dat', delimiter=',')
    # array of features
    X = np.array(data[:,:-1])
    # array of output class for corresponding feature set
    y = np.array(data[:,-1])
    # number of epochs
    epochs = 1000
    # learning rate
    learningRate = 0.009

    # get optimal parameters
    (W1,b1, W2, b2) = get_parameters(X, y,epochs,learningRate)

    # perform gradient checking
    gradient_checking(X, y, W1, b1,W2,b2)

    # print optimal parameters
    print("Optimal Parameters(weights and bias) are: \n", W1,b1,W2,b2, "\n")

    # calculate accuracy
    print("Accuracy:",get_accuracy(X,y,W1,b1,W2,b2),"%")

