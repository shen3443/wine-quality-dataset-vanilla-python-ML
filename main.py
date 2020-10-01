from get_data import GetData
from neural_network import NeuralNetwork
from results_plotter import PlotResults

path = 'winequality-white.csv' #select data file
hidden_activation_function = "sigmoid" #select activation function for hidden layer
test_size = 0.2 #portion of data reserved for testing
feature_scale_technique = "standardize"

if __name__ == '__main__':
    dataset = GetData(path) #get data from file
    dataset.feature_scale(feature_scale_technique)
    dataset.split_train_test(test_size) #divide data into training and test data
    
    n = NeuralNetwork(hidden_activation_function) #create neural network

    n.train(dataset.traindata, dataset.trainlabels) #train neural network with training data
    n.test(dataset.testdata, dataset.testlabels) #test the neural network with test data