import math

''' Sigmoid Function '''
def sigmoid(x):
    '''
    Activation function: compresses all (-inf, inf) => (0,1)
    Optimized with xavier initialization
    S(x) = 1 / (1 + e^-x)

    *May result in vanishing or exploding gradients. Function is near 
    horizontal for large inputs (positive or negative), so derivative
    is near 0
    '''
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return 1 - sigmoid(-x)

def d_sigmoid(x):
    # derivative of sigmoid function
    # S'(x) = (e^-x) / (1 + e^-x)^2 = S(x) * (1 - S(x))
    return sigmoid(x) * (1 - sigmoid(x))


''' ReLU Function '''
def relu(x):
    '''
    Activation function: Outputs x if x is positive, 0 if x is negative
    R(x) = {x if x > 0, 0 if x <= 0}

    *May result in dead neurons as negative inputs have a derivative of 0
    '''
    return max(0, x)

def d_relu(x):
    #derivative of ReLU function
    #R'(x) = {1 if x >= 0, 0 if x < 0}
    if x > 0:
        return 1
    else:
        return 0

''' leaky ReLU Function '''
def leakyrelu(x, a=0.01):
    '''
    Activation function: Outputs x if x is positive, alpha * x if x is negative
    where alpha is a small linear component, typically 0.01
    R(x) = {x if x > 0, a*x if x <= 0}
    '''
    return max((a * x), x)

def d_leakyrelu(x, a=0.01):
    #derivative of leaky ReLU function
    #R'(x) = {1 if x > 0, a if x <= 0}
    if x > 0:
        return 1
    else:
        return a

''' Softmax Function '''
def softmax(lst):
    '''
    Converts an input array lst = [x1,x2,x3...xn] into a probability
    distribution [p1,p2,p3...pn] where 0 > p > 1 for all p and the
    sum of all p = 1

    pn = S(xn) = e^xn / sum(e^x for all x)
    '''
    exp_lst = [math.exp(x) for x in lst]
    exp_sum = sum(exp_lst)
    soft = [x / exp_sum for x in exp_lst]
    return soft

def d_softmax(i, j, lst):
    #Derivative of softmax function
    #S' = S(i)*(s - S(j)) where s = 1 if i == j, s = 0 if i != j
    soft = softmax(lst)
    if i == j:
        s = 1
    else:
        s = 0
    return soft[i] * (s - soft[j])


def activation_derivative(activation_function, function_input, a=0.01):
    '''
    Calls the derivative of the apropriat activation_derivative

    *created to reduce cyclomatic complexity of train() method in
    NeuralNetwork class by breaking out the conditionals
    '''
    if activation_function == "sigmoid":
        return d_sigmoid(function_input)
    elif activation_function == "ReLU":
        return d_relu(function_input)
    elif activation_function == "leaky ReLU":
        return d_leakyrelu(function_input, a)
    else:
        raise Exception("Activation function not recognized (in backprop)")
