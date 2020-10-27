import math
from random import gauss


def initialize(technique, fan_in, fan_out, n_weights):
    '''
    Calls relevant weight-initialization function
    '''
    if technique == "xavier":
        return xavier(fan_in, fan_out, n_weights)
    elif technique == "he":
        return he(fan_in, n_weights)
    else:
        raise Exception("Initialization technique not recognized")


''' Xavier-normal Initialization '''


def xavier(n1, n2, w):
    '''
    Returns a gaussian-normal list with a mean of 0 and
    a Variance of sqrt( 2 / (fan-in + fan out))

    Used with Sigmoid and Softmax activations
    '''
    weights = []
    variance = math.sqrt(2 / (n1 + n2))
    sigma = math.sqrt(variance)  # standard deviation
    for i in range(w):
        weights.append(gauss(0, sigma))
    return weights


''' He-normal Initialization '''


def he(n, w):
    '''
    Returns a gaussian-normal list with a mean of 0 and
    a Variance of sqrt(2 / fan-in)

    Used with ReLU and leaky ReLU activations
    '''
    weights = []
    variance = math.sqrt(2/n)
    sigma = math.sqrt(variance)  # standard deviation
    for i in range(w):
        weights.append(gauss(0, sigma))
    return weights
