from activation import sigmoid, relu, leakyrelu


class Neuron:
    '''
    Neurons take a number of inputs: X = [x1,x2,x3...xn] and produce
    1 output: y via feedforward
    '''
    def __init__(self, weights, bias, activation):
        # Assign parameter inputs to class attributes
        self.__dict__.update({k: v for k, v in locals().items() if k != self})

    def feedforward(self, inputs):
        '''
        Returns output from given set of inputs

        y = A(f(X)) where: A is an activation function and
                            F(X) = x1*w1 + x2*w2... +xn*wn + b
                            for weights wn and bias b
        '''
        total = self.linearsum(inputs)
        if self.activation == "sigmoid":
            return sigmoid(total)
        elif self.activation == "ReLU":
            return relu(total)
        elif self.activation == "leaky ReLU":
            return leakyrelu(total)
        else:
            raise Exception("Activation function not recognized")

    def linearsum(self, inputs):
        '''
        Returns the output before being passed throught the activation
        function

        y = f(X) where:    f(X) = x1*w1 + x2*w2... +xn*wn + b
                            for weights wn and bias b
        '''
        if len(inputs) != len(self.weights):
            raise Exception("Number of inputs must equal number of weights")
        products = [i * w for i, w in zip(inputs, self.weights)]
        return (sum(products) + self.bias)

    def changeweight(self, newweight, index):
        '''
        Replaces old weight with new weight
        '''
        self.weights[index] = newweight

    def changebias(self, newbias):
        '''
        Replaces old bias with new bias
        '''
        self.bias = newbias
