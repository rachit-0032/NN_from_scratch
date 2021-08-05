# 2-layered Neural Network Functioning
 
import numpy as np

inputs = np.random.randint(-10, 10, (10, 4))              # Each input (3) has 4 features
weights_layer1 = np.random.randint(-5, 5, (3, 4))       # For each feature we have one weight, finally giving output to 3 perceptrons
biases_layer1 = np.random.randint(-20, 20, 3)           # each perceptron would have 1 bias variable

# Since we decided to have 3 perceptrons in the second layer (from weights), we go on building the second layer
weights_layer2 = np.random.randint(-5, 5, (1, 3))      # 1 output finally using 3 inputs
biases_layer2 = np.random.randint(-20, 20, 1)

# Doing the calculations
outputs_layer1 = np.dot(inputs, weights_layer1.T) + biases_layer1
outputs_layer2 = np.dot(outputs_layer1, weights_layer2.T) + biases_layer2

print("First Layer Outputs: ", outputs_layer1)
print("Second Layer Outputs: ", outputs_layer2)

## Had we created the matrix using lists, we would have to convert it into np.array() form else transpose would not have worked
## First layer would output a matrix of shape (training_samples, layer2_size)
## Second layer would output a matrix of shape (training_samples, output_size)


#########

# Generalisation
training_samples = 20
num_features = 10            
layer1_size = 5             # number of perceptrons in layer 1
layer2_size = 3             # number of perceptrons in layer 2
output_layer = 1            # number of output variables

inputs = np.random.randint(-10, 10, (training_samples, num_features))

weights_layer1 = np.random.randint(-5, 5, (layer1_size, num_features))
biases_layer1 = np.random.randint(-20, 20, layer1_size)

weights_layer2 = np.random.randint(-5, 5, (layer2_size, layer1_size))
biases_layer2 = np.random.randint(-20, 20, layer2_size)

weights_output = np.random.randint(-5, 5, (output_layer, layer2_size))
biases_output = np.random.randint(-20, 20, output_layer)

# Doing the calculations
outputs_layer1 = np.dot(inputs, weights_layer1.T) + biases_layer1
outputs_layer2 = np.dot(outputs_layer1, weights_layer2.T) + biases_layer2
outputs = np.dot(outputs_layer2, weights_output.T) + biases_output

print("Inputs: \n", inputs)
print("First Layer Outputs: \n", outputs_layer1)
print("Second Layer Outputs: \n", outputs_layer2)
print("Final Output: \n", outputs)