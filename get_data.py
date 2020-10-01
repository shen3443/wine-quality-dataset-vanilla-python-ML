import csv
import math

class GetData:
    '''
    Object that reads .csv file and stores usable data as properties
    Properties:
        self.path   =>      path of .csv File
        self.data   =>      input values*, two dimentional array of floats
        self.ytrue  =>      expected outputs, array of integers
    

    *input values have been adjusted such that:
        x <= (x - mean(x)) / mean(x) = x / mean(x) - 1
    '''
    def __init__(self, path):
        self.path = path
        self.data = []
        self.ytrue = []

        #check file format
        if self.path[-3:] != 'csv':
            raise Exception("Format not accepted. File must be .csv")

        with open(path) as csvfile: #open csv file
            datareader = csv.reader(csvfile) #read csv
            for row in datareader:
                rowlist = row[0].split(';')
                self.data.append(rowlist[:-1]) #inputs go to self.data
                self.ytrue.extend(rowlist[-1]) #expected outputs go to self.ytrue
        
        del self.data[0] #remove labels
        del self.ytrue[0:9] #remove labels

        #convert type to float for each data point, int for each label
        data_temp = [[float(x) for x in row] for row in self.data]
        ytrue_temp = [int(x) for x in self.ytrue]

        self.data = data_temp
        self.ytrue = ytrue_temp

        #classify labels
        #0 => bad, 1=> okay, 2 => good
        ytrue_temp = [] #clear ytrue_temp
        for y in self.ytrue:
            if y < 5:
                ytrue_temp.append(0) #wines with scores lower than 5 are "bad"
            elif y > 6:
                ytrue_temp.append(2) #wines with scores higher than 6 are "good"
            else:
                ytrue_temp.append(1) #wines with scores in between are "okay"
        
        self.ytrue = ytrue_temp


    def feature_scale(self, technique):
        data_invert = [[row[i] for row in self.data] for i in range(len(self.data[0]))]
        if technique == "standardize":
            feature_means = [sum(col)/len(col) for col in data_invert]
            feature_standard_deviations = [math.sqrt(sum([(x - mu )**2 for x in col])/len(col)) for mu, col in zip(feature_means, data_invert)]
            data_temp = [[(x - mu)/sigma for x, mu, sigma in zip(row, feature_means, feature_standard_deviations)] for row in self.data]
        elif technique == "normalize":
            feature_mins, feature_maxs = zip(*[(min(col),max(col)) for col in data_invert])
            fmins = [min(col) for col in data_invert]
            fmax = [max(col) for col in data_invert]
            data_temp = [[(x - x_min) / (x_max - x_min) for x, x_min, x_max in zip(row, feature_mins, feature_maxs)] for row in self.data]
        else:
            raise Exception("Feature Scaling technique not recognized")
        self.data = data_temp

    def split_train_test(self, test_size=0.2):
        '''
        For both the input and label arrays, divides the data into two seperate
        arrays, one for training and one for testing. Parameter test_size 
        represents the portion of data reserved for testing (ie. 0.2 means that
        20% of the data is reserved).
        '''
        point = int(len(self.data) * (1-test_size)) + 1
        self.traindata = self.data[:point]
        self.testdata = self.data[point:]
        self.trainlabels = self.ytrue[:point]
        self.testlabels = self.ytrue[point:]
