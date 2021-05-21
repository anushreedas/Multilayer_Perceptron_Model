import numpy as np
import matplotlib.pyplot as plt
"""
q2.py

This program fits the previous MLP network to the MNIST training set,
where parameter optimization operates with mini-batches and
estimates accuracy by using the validation/development dataset given.
Finally, the model is tested on test set by checking accuracy of the model for every class 

@author: Anushree Sitaram Das (ad1707)
"""


def create_mini_batch(X,y,batchSize):
    """
    Divides given dataset into mini batches

    :param X:           Input Features
    :param y:           Output Class
    :param batchSize:   size of each batch
    :return:            mini batches
    """
    # stores the mini batches
    X_mini_batches = []
    Y_mini_batches = []

    # total number of batches
    n_minibatches = X.shape[0] // batchSize

    # divide dataset into small batches of equal sizes
    for i in range(n_minibatches):
        X_mini = X[i * batchSize:(i + 1) * batchSize, :]
        Y_mini = y[i * batchSize:(i + 1) * batchSize, :]
        X_mini_batches.append(X_mini)
        Y_mini_batches.append(Y_mini)
    # last batch of leftover data
    if X.shape[0] % batchSize != 0:
        X_mini = X[n_minibatches * batchSize:X.shape[0]]
        Y_mini = y[n_minibatches * batchSize:y.shape[0]]
        X_mini_batches.append(X_mini)
        Y_mini_batches.append(Y_mini)

    return X_mini_batches,Y_mini_batches


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
    return 1 / (1 + np.exp(-x))


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
    z1 = np.dot(X[:,0], W1)+np.dot(X[:,1], W1) + b1
    # activation at hidden layer
    a1 = sigmoid(z1)
    # (output from hidden layer . weights 2) + bias 2
    z2 = np.dot(a1, W2) + b2
    # activation at output layer
    a2 = softmax(z2)

    # regularization term
    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2))) * (lamb / (2 * N))
    # cross-entropy loss with regualrization
    loss = - (np.sum(y * np.log(a2))) / N + L2_regularization_cost

    # backpropagation
    a2_delta = (a2 - y) / N  # w2
    z1_delta = np.dot(a2_delta, W2.T)
    a1_delta = z1_delta * sigmoid_derivative(a1)  # w1

    # gradients for all parameters
    gradW2 = (1 / N) *((np.dot(a1.T,a2_delta) + (lamb * W2)))
    gradb2 = np.sum(a2_delta, axis=0, keepdims=True)
    gradW1 = (1 / N) * ((np.dot(X[:,0].T, a1_delta)+np.dot(X[:,1].T, a1_delta)) + (lamb * W1))
    gradb1 = np.sum(a1_delta, axis=0)


    return loss, gradW1, gradb1, gradW2, gradb2


def vertor_to_matrix(y,c):
    """
    Converts output class vector to matrix.
    Ex:
    vector = [0,1,2,1]
    matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0]]
    :param y:   vector
    :param c:   total number of classes
    :return:    matrix
    """
    mat = (np.arange(c) == y[:, None]).astype(float)
    mat = mat.reshape(-1,mat.shape[-1])
    return mat


def predict(X, W1,b1,W2,b2):
    """
    Predict class for given vector of features

    :param X: Input features
    :param W: Weights
    :param b: Bias
    :return:  Class of given input features
    """
    # feedforward
    # (input features . weights 1) + bias 1
    z1 = np.dot(X[:, 0], W1) + np.dot(X[:, 1], W1) + b1
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
    :param W: Weights
    :param b: Bias
    :return:  Accuracy in percentage
    """

    pred = (predict(X,W1,b1,W2,b2))
    sum = 0
    for i in range(len(pred)):
        if pred[i]==y[i][0]:
            sum+=1
    accuracy = sum/(float(len(y)))
    return accuracy*100
    return 0

def get_class_accuracy(X,y,W1,b1,W2,b2):
    """
    Calculates accuracy of predictions for the given model
    :param X: Input features
    :param y: Output class
    :param W: Weights
    :param b: Bias
    :return:  Accuracy in percentage
    """
    pred = predict(X, W1, b1, W2, b2)
    labels = np.unique(y_train)
    accuracy = []
    for label in labels:
        print('Checking for class: ',label)
        d = np.count_nonzero(y == label)
        n = 0
        incorrect_found = False
        correct_found = False
        for i in range(len(pred)):
            if y[i][0]==label:
                if pred[i] == y[i][0]:
                    n+=1
                    if not correct_found:
                        correct_found = True
                        print('Predicted Label: ', pred[i])
                        print('Actual Label: ', y[i, :])
                        plt.imshow(X[i, :], cmap=plt.get_cmap('gray'))
                        plt.savefig('data/correct' + str(label) + '.jpg')
                        plt.show()
                else:
                    if not incorrect_found:
                        incorrect_found = True
                        print('Predicted Label: ',pred[i])
                        print('Actual Label: ', y[i,:])
                        plt.imshow(X[i,:], cmap=plt.get_cmap('gray'))
                        plt.savefig('data/incorrect'+str(label)+'.jpg')
                        plt.show()
        accuracy.append(n/d)
    return accuracy


def train_model(X_train, y_train,X_test,y_test,epochs,batchSize,learningRate,lamb,earlyStopping):
    """
      Calculates optimal values for weights and bias via regression

      :param X_train:          Input features for training model
      :param y_train:          Output class for training model
      :param X_test:           Input features for evaluating model
      :param y_test:           Output features for evaluating model
      :param epochs:           Number of epochs
      :param batchSize:        Size of each batch
      :param learningRate:     Learning rate for this model
      :param lamb:             Lambda value
      :return:                 Optimal weights and bias
    """

    # total number of classes
    num_labels = len(np.unique(y_train))
    # number of units in the hidden layer
    num_nodes = 20
    d = np.sqrt(1 / (X_train.shape[1] + num_labels))
    # initialize Weights 1
    W1 = np.random.randn(X_train.shape[1], num_nodes) * d
    # initialize Bias 1
    b1 = np.zeros(num_nodes) * d

    # initialize Weights 2
    W2 = np.random.randn(num_nodes, num_labels) * d
    # initialize Bias 2
    b2 = np.zeros(num_labels) * d

    # stores loss for each epoch for training
    train_loss = []
    # stores accuracy for each epoch for training
    train_accuracy =[]
    # stores loss for each epoch for validation
    test_loss = []
    # stores accuracy for each epoch for validation
    test_accuracy = []

    for i in range(0, epochs):
        if i%100 == 0:
            print('############# Running epoch ',i,' / ',epochs,'..')
        # get mini batches of given training dataset
        X_mini_batches,Y_mini_batches = create_mini_batch(X_train, y_train, batchSize)

        # keep track of last previous value parameter
        lastW1 = W1
        lastb1 = b1
        lastW2 = W2
        lastb2 = b2
        for X_mini, Y_mini in zip(X_mini_batches,Y_mini_batches):

            # convert output vector to matrix
            y_mini = vertor_to_matrix(Y_mini,num_labels)

            # get gradient descents for weights and bias
            loss,  gradW1, gradb1, gradW2, gradb2 = loss_function(W1,b1,W2,b2, X_mini, y_mini,lamb)

            # update weights and bias
            W1 = W1 - (learningRate * gradW1)
            b1 = b1 - (learningRate * gradb1)
            W2 = W2 - (learningRate * gradW2)
            b2 = b2 - (learningRate * gradb2)


        # convert output vector to matrix
        Y_train = vertor_to_matrix(y_train,num_labels)
        # get loss for current weights and bias for training set
        loss,  gradW1, gradb1, gradW2, gradb2 = loss_function(W1,b1,W2,b2, X_train, Y_train)
        train_loss.append(loss)
        # get accuracy for current weights and bias for training set
        accuracy = get_accuracy(X_train, y_train,W1,b1,W2,b2)
        train_accuracy.append(accuracy)

        # convert output vector to matrix
        Y_test = vertor_to_matrix(y_test,num_labels)
        # get loss for current weights and bias for validation set
        loss,  gradW1, gradb1, gradW2, gradb2 = loss_function(W1,b1,W2,b2, X_test,Y_test)
        test_loss.append(loss)
        # get accuracy for current weights and bias for validation set
        accuracy = get_accuracy(X_test,y_test, W1,b1,W2,b2)
        test_accuracy.append(accuracy)

        # if loss starts to increase revert the parameter update
        if len(test_loss) > earlyStopping and test_loss[-1]>test_loss[-2]:
            W1 = lastW1
            b1 = lastb1
            W2 = lastW2
            b2 = lastb2
            break

    # Plot losses for training and validation datasets
    plt.plot(train_loss, '--',alpha = 1.0)
    plt.plot(test_loss, alpha = 0.5)
    plt.savefig('losses_q2.png')
    plt.show()

    # Plot accuracy for training and validation datasets
    plt.plot(train_accuracy, '--',alpha = 1.0)
    plt.plot(test_accuracy, alpha = 0.5)
    plt.savefig('accuracy_q2.png')
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


    # load training data
    # array of features
    X = np.load('train_images_array.npy')
    # array of output class for corresponding feature set
    y = np.load('train_labels_array.npy')

    # shuffle dataset
    s = np.arange(X.shape[0])
    X = X[s]
    y = y[s]

    # calculate split index
    split_index = (int)(0.8 * X.shape[0])
    # split it into training and validation dataset
    X_train, X_val = X[:split_index, :], X[split_index:, :]
    y_train, y_val = y[:split_index, :], y[split_index:, :]

    # load testing data
    # array of features
    X_test = np.load('test_images_array.npy')
    # array of output class for corresponding feature set
    y_test = np.load('test_labels_array.npy')

    # number of epochs
    epochs = 600
    # mini batch size
    batchSize = 10
    # learning rate
    learningRate = 0.0001 #0.001
    lamb = 0.01
    # check loss for early stopping after the following number of iterations
    earlyStopping = 100


    print('Training..')
    # get optimal parameters
    (W1,b1, W2, b2) = train_model(X_train, y_train,X_val,y_val,epochs,batchSize,learningRate,lamb,earlyStopping)

    # # perform gradient checking
    # gradient_checking(X_train, y_train,W1,b1, W2, b2)

    # print optimal parameters
    # print("Optimal Parameters are: \n", W1,b1, W2, b2, "\n")

    print('Testing..')

    class_accuracies = get_class_accuracy(X_test,y_test,W1,b1, W2, b2)
    labels = np.unique(y_train)
    plt.bar(labels,class_accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.xticks(np.arange(len(labels)), labels)
    plt.savefig('histrogram.png')
    plt.show()

