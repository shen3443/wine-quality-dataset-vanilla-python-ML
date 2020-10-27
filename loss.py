import math


def mseLoss(yPred, yTrue):
    '''
    Takes two arrays and outputs the mean squared error loss (MSE Loss)

    L = Sum((yTrue-yPred)^2)

    L' = -2 * (yTrue * yPred)
    '''
    total = 0.0  # sum of squared error
    for i in range(len(yPred)):
        total += ((yTrue[i] - yPred[i])**2)
    return total / len(yPred)  # mean of squared error


def crossEntropyLoss(soft, yTrue):
    '''
    Takes an array of softmax outputs and an int value of the correct y value
    and outputs the cross entropy loss

    L = -ln(pc)     where ln is the natural log and pc is the probability
                    estimated for the correct case

    L' = {  pi - 1  if i = c
            0       if i != c

                    where i is each case in softmax, c is the correct case,
                    and pi is the probability of case i

    '''
    return math.log(soft[yTrue]) * -1
