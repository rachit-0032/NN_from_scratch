# Single Perceptron Working

import sys
import numpy as np
import matplotlib


# Version of the Packages being used.
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Matplotlib: ", matplotlib.__version__)


# A single perceptron
inputs = [1.2, 5.1, 2.1]        # 3 inputs to the single neuron.
weights = [3.1, 2.1, 8.7]       # 1 weight for each of the input.
bias = 3
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
ans = np.dot(inputs, weights) + bias

assert(output == ans)           # ensures that both things are doing the same thing.
print(output)

## Every neuron has a single bias, while every input as a separate weight.

### 
# In case there are more inputs and more outputs, the use of matrices using numpy becomes 
# important. Wrong placements of matrices can either result in shape errors, or a completely
# different result which is difficult to debug.