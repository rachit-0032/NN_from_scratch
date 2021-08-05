import numpy as np

np.random.seed(0)                                                   # ensures that the random output remains same for every run of the file

# Creates layers and allows forwards propagation
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)     # so that transposing is not required at any stage
        self.biases = np.zeros((1, n_neurons))

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# User-Input
n_samples = int(input("Number of Training Samples: "))
n_features = int(input("Number of Features for each sample: "))


# Define Layers
inputs = np.random.randint(-10, 10, (n_samples, n_features))
layer1 = Layer_Dense(n_features, 16)                                # n_features*16 --> 16 neurons in the first layer        
layer2 = Layer_Dense(16, 32)                                        # 32 neurons in the second layer
layer3 = Layer_Dense(32, 16)
layer4 = Layer_Dense(16, 4)
layer5 = Layer_Dense(4, 1)                                          # finally giving out one output for each training sample


# Forward Propagation
layer1.forward_pass(inputs)
layer2.forward_pass(layer1.output)
layer3.forward_pass(layer2.output)
layer4.forward_pass(layer3.output)
layer5.forward_pass(layer4.output)

print("Final Output: \n", layer5.output)

## Multiplying the weights with a number between -1 & 1 ensures that the output doesn't explode by constantly multiplying by a larger number
## If the network comes out to be dead i.e. output as 0 always, then the biases need to be initialised with a non-zero value so that outputs to each neuron changes a bit.