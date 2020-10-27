'''
Author: Chris Isacs

For more info visit:
https://github.com/shen3443/wine-quality-dataset-vanilla-python-ML
'''

from get_data import GetData
from get_input import GetInput
from neural_network import NeuralNetwork

# data file
PATH = 'winequality-white.csv'


def main():
    # initialize class to handle user input
    user = GetInput()

    # get data from file
    dataset = GetData(PATH)
    # scale features
    dataset.feature_scale(user.get_feature_scale_technique())
    # divide data into training and test data
    dataset.split_train_test(user.get_test_size())

    # create neural network
    n = NeuralNetwork(user.get_activation())

    # train neural network with training data
    n.train(dataset.traindata, dataset.trainlabels)
    # test the neural network with test data
    n.test(dataset.testdata, dataset.testlabels)


if __name__ == '__main__':
    main()
