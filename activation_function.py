# Activation Function Application

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

## Currently, we are only working on the forward part and thus, the target variables are not actually needed. We want to figure out out the choice of activation
# functions impacts the final result.

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases


## ReLU i.e. Rectified Linear Unit is an Activation Function that gives output as 0 for all negative inputs, while the value itself in all other cases. 
## In contrast to a simple linear unit, this allows us use the non-linearity of the data and is computationally cheaper than sigmoid function.
## Sigmoid function has a problem of vanishing gradients and thus, ReLU acts as a complete package for being a great activation function.
## An activation function can be useful to bound the output, to extract non-linearities, amplify the granuality of the difference in outputs at times, etc.
class ActivationReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)


## Softmax is useful in the output layer of the NN to get the result in terms of probabilities. The largest of all would mean that the input is most likely to be 
# classified as that target.
class ActivationSoftmax:
    def forward(self, input):
        # np.sum(np.exp(input), axis=1, keepdims=True)                                # keepdims = True, allows us to get the result in the original shape
        exp_values = np.exp(input) - np.max(input, axis=1, keepdims=True)
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)


## We subtract the maximum of all feature values to ensure that the values are finally <= 0, which when exponentiated, would give a result in [0,1], thus 
# saving us the trouble of exploding due to multiple multiplications.

layer1 = Layer_Dense(len(X[0]), 10)
activation1 = ActivationReLU()

layer2 = Layer_Dense(10, 5)
activation2 = ActivationReLU()

layer_output = Layer_Dense(5, 4)
activation_output = ActivationSoftmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer_output.forward(layer2.output)
activation_output.forward(layer_output.output)

print("Feed Forward Output: \n", activation_output.output)