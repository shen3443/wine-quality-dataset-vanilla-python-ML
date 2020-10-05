'''
Author: Chris Isacs

For more info visit:
https://github.com/shen3443/wine-quality-dataset-vanilla-python-ML
'''

from get_data import GetData
from get_input import GetInput
from neural_network import NeuralNetwork

PATH = 'winequality-white.csv' #data file

def main():
    user = GetInput() #initialize class to handle user input

    dataset = GetData(PATH) #get data from file
    dataset.feature_scale(user.get_feature_scale_technique()) #scale features
    dataset.split_train_test(user.get_test_size()) #divide data into training and test data
    
    n = NeuralNetwork(user.get_activation()) #create neural network

    n.train(dataset.traindata, dataset.trainlabels) #train neural network with training data
    n.test(dataset.testdata, dataset.testlabels) #test the neural network with test data

if __name__ == '__main__':
    main()
