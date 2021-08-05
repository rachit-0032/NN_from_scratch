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

class ActivationReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)

class ActivationSoftmax:
    def forward(self, input):
        # np.sum(np.exp(input), axis=1, keepdims=True)                                  # keepdims = True, allows us to get the result in the original shape
        exp_values = np.exp(input) - np.max(input, axis=1, keepdims=True)
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)
        return np.mean(sample_losses)                                                   # This is the overall loss averaged over all the samples

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)                                  # since 0 probability of the correct class would result in infinite loss, we clip the values

        ## Now, if the targets are given in the one-hot encoded form (len == 2), we need to treat them differently, while if we are given a single long array of 
        # targets, we need to ensure we pick the right probabilities from the output layer's output.

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]              # for every row, it takes out the probability of the correct instance as per the target mentioned in y_true
        else:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)                 # we do sum, because other values have been converted to zero since this is one-hot encoding structure

        return -1 * np.log(correct_confidences)



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

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation_output.output, y)

print("Feed Forward Output: \n", activation_output.output)
print("\n First Iteration Loss: ", loss)

## This loss value is quite big because currently, the input feels random to the NN and thus the probabilities are more or less equally distributed.
## This loss now needs to be communicated to each neuron mathematically so that it knows how to get the right thing done.