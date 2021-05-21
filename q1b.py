import numpy as np
import matplotlib.pyplot as plt

"""
q1b.py

This program train the previous single hidden layer MLP on the data and then plot its decision boundary.

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
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


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
    loss = - np.sum(y * np.log(a2)) / N

    a2_delta = (a2 - y) / N  # w2
    z1_delta = np.dot(a2_delta, W2.T)
    a1_delta = z1_delta * sigmoid_derivative(a1)  # w1

    # gradients for all parameters
    gradW2 = (1 / N) * (np.dot(a1.T, a2_delta)) + (lamb * W2)
    gradb2 = np.sum(a2_delta, axis=0, keepdims=True)
    gradW1 = (1 / N) * (np.dot(X.T, a1_delta)) + (lamb * W1)
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
    num_units = 25
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
    accuracy = []

    for i in range(0, epochs):
        # get loss and gradient descents for weights and bias
        loss,  gradW1, gradb1, gradW2, gradb2 = loss_function(W1,b1,W2,b2, X, y,0.1)
        losses.append(loss)
        # update weights and bias
        W1 = W1 - (learningRate * gradW1)
        b1 = b1 - (learningRate * gradb1)
        W2 = W2 - (learningRate * gradW2)
        b2 = b2 - (learningRate * gradb2)

        accuracy.append(get_accuracy(X, Y, W1, b1, W2, b2))

    # Plot Losses
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss over epochs")
    plt.savefig('losses_q1b.png')
    plt.show()
    plt.plot(accuracy)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Accuracy over epochs")
    plt.savefig('acc_q1b.png')
    plt.show()

    return W1,b1,W2,b2


def predict(X, W1,b1,W2,b2):
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
    # print(pred.shape)
    sum = 0
    for i in range(len(pred)):
        if pred[i]==y[i]:
            sum+=1
    accuracy = sum/(float(len(y)))
    return accuracy*100


def gradient_checking(X, Y, W1, b1, W2, b2):
    """
    Performs gradient-checking.

    :param X: Input features
    :param y: Output class
    :param W: Weights
    :param b: Bias
    :return:  None
    """
    # convert output vector to matrix
    y = vertor_to_matrix(Y)
    epsilon = 10e-4

    # check approximations of derivatives for weights 1
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


def plot_decision_boundary(X,y,  W1, b1, W2, b2):
    """
    Plots decision boundary for the given dataset using the optimal parameters given
    :param X: Input features
    :param W: Weights
    :param b: Bias
    :return:  None
    """

    plt.grid()
    plt.xlabel('feature 1', size=20)
    plt.ylabel('feature 2', size=20)

    # get both feature values for plotting in 2d place
    feature1 = X[:, 0]
    feature2 = X[:, 1]

    # assign color for each input according to its output class
    colors = []
    for c in y:
        if c == 0:
            colors.append('green')
        else:
            if c == 1:
                colors.append('red')
            else:
                colors.append('blue')

    # plot features
    plt.scatter(feature1, feature2, s=5, color=colors)

    # build heatmap
    # find min and max values of both features
    x_min, x_max = feature1.min() - 1, feature1.max() + 1
    y_min, y_max = feature2.min() - 1, feature2.max() + 1

    # Predict class for all combinations of values of both features
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    Xnew = np.c_[xx.ravel(), yy.ravel()]
    pred = predict(Xnew, W1, b1, W2, b2)
    pred = pred.reshape(xx.shape)

    plt.contourf(xx,yy,pred,alpha=0.5)

    plt.savefig('decision_boundary_q1b.png')
    plt.show()


if __name__ == "__main__":
    # load data
    data = np.genfromtxt('spiral_train.dat', delimiter=',')
    # array of features
    X = np.array(data[:,:-1])
    # array of output class for corresponding feature set
    y = np.array(data[:, -1])
    # number of epochs
    epochs =  1000
    # learnig rate
    learningRate = 0.005

    # get optimal parameters
    (W1,b1, W2, b2) = get_parameters(X, y, epochs,learningRate)

    # perform gradient checking
    gradient_checking(X, y, W1, b1, W2, b2)

    # print optimal parameters
    print("Optimal Parameters(weights and bias) are: \n", W1, b1, W2, b2, "\n")

    # calculate accuracy
    print("Accuracy:", get_accuracy(X, y, W1, b1, W2, b2), "%")

    plot_decision_boundary(X, y, W1, b1, W2, b2)
