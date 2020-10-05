from activation import activation_derivative, softmax, d_softmax
from loss import crossEntropyLoss
from weight_initialization import initialize
from neuron import Neuron
  

class NeuralNetwork:
    '''
    Neural network initializes by creating two layers, a hidden layer and
    a softmax layer
    '''
    def __init__(self, hiddenActivation="sigmoid"):
        self.hiddenActivation = hiddenActivation #activation function for hidden layer

        #Select appropriate weight initialization for activation function
        if self.hiddenActivation == "sigmoid":
            init_technique = "xavier"
        elif self.hiddenActivation in ["ReLU", "leaky ReLU"]:
            init_technique = "he"
        else:
            raise Exception("Activation for hidden layer not recognized")
        
        #create hidden layer with 5 neurons
        self.hidden = [Neuron(initialize(init_technique, 11, 3, 11), 0, self.hiddenActivation) for i in range(5)]
        
        #create a softmax layer with 3 neurons
        self.soft = [Neuron(initialize("xavier", 5, 1, 5), 0, None) for i in range(3)]


    def feedforward(self, inputs):
        '''
        Passes inputs forward through neural network, returns an array of
        probabilities

        Input Layer => Hidden Layer => Softmax Layer
        '''
        #feedforward inputs through hidden layer
        hidden_outputs = [node.feedforward(inputs) for node in self.hidden]

        #feedforward hidden layer outputs through softmax layer
        return softmax([node.linearsum(hidden_outputs) for node in self.soft])
  

    def train(self, data, yTrues, learnRate=0.1, epochs=1000, checkRate=50):
        '''
        Uses backpropagation to calculate the partial derivatives (gradience)
        of Loss in regard to each weight and bias, then uses Stochastic Gradient
        Descent (SGD) to adjust each weight and bias such that:

            w <= w-lr*dLdw where:   w is a weight or bias,
                                    lr is the learn rate (typically 0.1)
                                    dwdL is the partial derivative Loss in
                                    regards to that weight or bias
        
        Each epoch represents one run through the entire dataset, and after every
        50 epochs (or however many is set by the check rate) the program will run
        a feedforward and print the Epoch number, the Cross Entropy Loss and the 
        Accuracy (percent of correct guesses)
        '''
        for epoch in range(epochs): #a runthrough of the entire data set
            for x, yTrue in zip(data, yTrues): #a runthrough of one row of data

                #execute a feedforward, storing linear sums (results of linear function in neuron)
                hidden_outputs = []
                hidden_totals = []
                for node in self.hidden:
                    hidden_totals.append(node.linearsum(x))
                    hidden_outputs.append(node.feedforward(x))

                soft_totals = [node.linearsum(hidden_outputs) for node in self.soft]
                soft_outputs = softmax(soft_totals)
                
                #partial derivatives
                #partial L / partial softout for case c (yTrue)
                dL_dsoc = -1 / soft_outputs[yTrue]
                
                #Update softmax layer
                j = 0 #counter for index of current neuron in self.soft
                for node in self.soft:
                    dsoc_dt = d_softmax(yTrue, j, soft_totals) #partial softout for case c / partial total

                    #Update weights
                    for w in range(len(node.weights)):
                        dt_dw = hidden_outputs[w] #partial total / partial weight
                        partial_derivative = dL_dsoc * dsoc_dt * dt_dw #partial loss / partial weight
                        newweight = node.weights[w] - learnRate * partial_derivative #SGD
                        node.changeweight(newweight, w) #update weight

                    #Update biases
                    partial_derivative = dL_dsoc * dsoc_dt #partial loss / partial bias
                    newbias = node.bias - learnRate * partial_derivative #SGD
                    node.changebias(newbias) #update bias
                    j += 1 #increment counter
                    h = 0 #counter for index of current neuron in self.hidden

                    for hnode in self.hidden:
                        dt_dh = node.weights[h] #partial total(softmax layer) / partial h

                        for w in range(len(hnode.weights)):
                            dh_dw = x[w] * activation_derivative(self.hiddenActivation,hidden_totals[h]) #partial h / partial w
                            partial_derivative = dL_dsoc * dsoc_dt * dt_dh * dh_dw #partial loss / partial w
                            newweight = hnode.weights[w] - learnRate * partial_derivative #SGD
                            hnode.changeweight(newweight, w) #update weight
                            
                        dh_db = activation_derivative(self.hiddenActivation, hidden_totals[h]) #partial h / partial bias
                        partial_derivative = dL_dsoc * dsoc_dt * dt_dh * dh_db #partial loss / partial bias
                        newbias = hnode.bias - learnRate * partial_derivative #SGD
                        hnode.changebias(newbias) #update bias
                        h += 1 #increment counter
        
            #Run a feedforward on the data and print an update to the console with the epoch, avg. loss, accuracy
            if epoch % checkRate == 0:
                self.test(data, yTrues, True, epoch)

    def test(self, data, yTrues, duringtrain=False, epoch=None):
        '''
        Runs a feedforward through data and prints an assessment to the consol

        Assessment gives average loss and accuracy, as well as the epoch if the
        test is occuring during training
        '''
        #Run feedforward
        yPreds = [self.feedforward(x) for x in data]

        #Calculate losses for each row
        losses = [crossEntropyLoss(soft, y) for soft, y in zip(yPreds, yTrues)]

        #Calculate average loss (mean)
        avgLoss = sum(losses) / len(losses)
        correct = 0 #counter for correct guesses

        #Find case with highest probability from softmax (the model's guess)
        yPredictions = [soft.index(max(soft)) for soft in yPreds]

        #count how many guesses were correct
        for y, yhat in zip(yTrues, yPredictions):
            if y == yhat:
                correct += 1
                
        #Calculate accuracy as a percent of correct guesses
        accuracy = 100 * correct / len(yPreds)

        #Print results to console
        if duringtrain:
            print("Epoch: %d    Average Loss: %.3f    Accuracy %.2f%%" % (epoch, avgLoss, accuracy))
        else:
            print("Average Loss: %.3f    Accuracy %.2f%%" % (avgLoss, accuracy))
