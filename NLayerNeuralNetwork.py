import numpy as np
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

# A class to create a N layer neural network that uses the relu function
# for all hidden layers and then a sigmoid function for the output layer.


# Initialise W and b. W and b need a 2D matrix for each layer of the NN.
def initialise_parameters(layer_dims):

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(
            layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


# Calculate the Z value based on activations from previous layer, weights from the current layer and
# the bias of the current layer.
def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


# Compute A using the layer's specified activation function.
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


# Call the linear forward and activation functions for each layer, using the relu function for all hidden
# layers and then the sigmoid for the output layer to give us a yhat between 0 and 1.
def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


# Compute the cross-entropy cost using a complex function given by Courserra.
def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = (-1/m) * (np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL))))
    cost = np.squeeze(cost)

    return cost


# Compute the linear part of the backward propagation for one layer using complex
# derivatives given by Courserra.
# Calculate dW, db and dA_prev using dZ.
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# Compute dZ using dA and the inverse of the given activation function.
# Then use dZ to calculate dA_prev, dW and db.
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_function = cache

    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])

    return dA_prev, dW, db


# Compute the derivatives by calling linear_activation_backward() for each layer
# with the appropriate activation function.
def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    # from L-2 to 0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# Update the parameters (W, b) for each layer using the derivatives calculated
# in the back propagation and the given learning rate.
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * grads["dW" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * grads["db" + str(l)])

    return parameters


# Function given by Courserra which predicts the results of the NN.
# Returns the predictions made by the NN for X.
def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p
