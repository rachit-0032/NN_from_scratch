# Multi-Perceptron Working
# Below is an example of 4 perceptrons interacting with 3 output perceptrons.

import numpy as np

# Using matrices to do the work
inputs = np.random.randint(0, 10, 4)        # 4 inputs.
weights = np.random.randint(0, 10, (3,4))   # 3*4 weight matrix.
biases = np.random.randint(0, 10, (1,3))    # 1 bias term for each perceptron.

## For every input, we would have an associated weight. Here, we have 3 inputs,
## and thus the dimension of the matrices need to be seen carefully.

## outputs = np.dot(inputs, weights) + biases   # This would call in for shape error.
outputs = np.dot(inputs, weights.T) + biases
ans = np.dot(weights, inputs) + biases
assert((outputs == ans).all())                  # ensuring that all elements are equal.

print(outputs)

###
# It is better to have 'weights' as the first argument as the latter allows us to visualise
# things better, since the number of rows of the weight matrix define the number of outputs
# that we seek to have from the multiplication, in this case 3.