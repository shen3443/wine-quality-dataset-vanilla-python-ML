# wine-quality-dataset-vanilla-python-ML
A feedforward neural network to predict wine quality based on a number of scientific factors. 

**NOTE:** This is purely an educational project. This is neither an efficient nor realistic neural network for practical use.

When trying to learn about Neural Networks, I found a common theme where articles would articulate concepts well, but the code would be very difficult to follow because a significant number of complex steps would be executed with one very simple command on one line of code using powerful ML libraries like tensorflow or keras. While these libraries are very important for practical applications of ML, I wanted to build a neural network in vanilla python with (*hopefully*) easier-to-follow code for ML beginners. Undertaking this project helped me learn a lot about the fundamentals of feedforward Neural Networks.



The data is from: https://archive.ics.uci.edu/ml/datasets/wine+quality

citation:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


## Files
#### main.py
`main.py` allows the user to set a number of variables that act as 'settings', such as `hidden_activation_function` (which activation function to use for the hidden layer) or `feature_scale_technique` (whether to normalize or standardize the input data).

Then `main.py` creates instances of classes from the other files and calls methods on them to collect and prep the data, train the model on the data and finally test the model.

#### neuron.py
`neuron.py` contains the class `Neuron`. Neurons contain weights and biases and can perform methods like `feedforward()`, `linearsum()` (essentially a feedforward without the activation function, which is usefull in backprop), `changeweight()` and `changebias()`. Neurons act as the building blocks of the Neural Network.

#### neural_network.py
`neural_network.py` contains the class `NeuralNetwork`. The neural network consists of arrays (*layers*) of Neurons and can perform methods like `feedforward()`, `train()` and `test()`. `neural_network.py` is the largest and most dense/complex python file in the folder, largly because of the `train()` method (which deals with backprop).

#### get_data.py
`get_data.py` contains the class `GetData`, which handles reading the .csv file, extracting and cleaning the data, scaling the features, and splitting the data into train and test data. It does a lot of this when it is initialized, but is aided by the methods `feature_scale()` (which can normalize or standardize the data, depending on what is passed to the `technique` parameter) and `split_train_test()`.

#### activation.py
`activation.py` contains functions to execute a variety of activation functions and their derivatives. The `d_` prefix is used to denote functions that represent derivatives (ie. `d_sigmoid()` is the derivative of the Sigmoid function). Supported activations include the Sigmoid Function (`sigmoid()` and `d_sigmoid`), ReLU (`relu()` and `d_relu()`), and leaky ReLU (`leakyrelu()` and `d_leakyrelu()`). The softmax function and it's derivative (`softmax()` and `d_softmax()`) are also defined in `activation.py`. Finally the helper function `activation_derivative()` is defined to conditionally call the appropriate derivative function during backprop (to simplify the `train()` method of `NeuralNetwork` and reduce its cyclomatic complexity).

#### weight_initialization.py
`weight_initialization.py` contains functions related to initializing a neurons weights. Xavier-normal initializaion (`xavier()`) and He-normal initialization (`he()`) are supported. `weight-initialization.py` also has a helper function `initialize()`, which conditionally calls the relevant weight initialization function.

#### loss.py
`loss.py` contains functions related to measuring loss. Mean squared error loss (`mseLoss()`) and cross entropy loss (`crossEntropyLoss()`) are both supported.

*note:* The current version of the model uses a softmax layer, so MSE loss is not applicable. Former versions of the model had different approaches for which MSE loss was required, and I've left it in to 1) add flexibility for anyone who wants to play around with the model and 2) display what an implementation of MSE loss might look like.

#### winequality-white.csv
The .csv file containing the data, as downloaded from https://archive.ics.uci.edu/ml/datasets/wine+quality




## To Do:
### Short Term
- [ ] Finish README
- [ ] Reveiw code, make sure latest versions are represented, outdate/irrelevant code/comments are removed
- [ ] Upload analysis of data and model
- [ ] Replace hardcoded 'settings' in `main.py` with command line user inputs

### Long Term
- [ ] Detailed, long form explanation of all concepts & code
- [ ] Design, generate and upload graphs (Matplotlib?) & graphics to aid analysis
- [ ] Comparitive model built using ML library (could be an interesting comparison to see how much more efficient/effective it is vs vanilla python).
