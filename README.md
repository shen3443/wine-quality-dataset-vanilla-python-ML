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
'<main.py>' allows the user to set a number of variables that act as 'settings', such as '<hidden_activation_function>' (which activation function to use for the hidden layer) or '<feature_scale_technique>' (whether to normalize or standardize the input data).

Then '<main.py>' creates instances of classes from the other files and calls methods on them to collect and prep the data, train the model on the data and finally test the model.

#### neuron.py
'<neuron.py>'


## To Do:
### Short Term
-[ ] Finish README
-[ ] Reveiw code, make sure latest versions are represented
-[ ] Upload analysis of data and model

### Long Term
-[ ] Detailed, long form explanation of all concepts & code
-[ ] Design, generate and upload graphs (Matplotlib?) & graphics to aid analysis
-[ ] Comparitive model built using ML library (could be an interesting comparison to see how much more efficient/effective it is vs vanilla python).
