# Understanding Loss Function

###
# In the activation function, we explored that in the output layer, we tend to use the softmax activation function which gives us the probabilities or the 
# confidence of putting a certain target in a specific class. If we have low confidence, this should have a high penalty, while a high confidence level would 
# indicate that the penalty or the 'loss' is low. This is the reason why 'Categorical Cross Entropy' is quite popular as the log function makes low confidence 
# output have a high penalty. This loss is to be minimised using the information of the actual variables and shall be backpropagated through some mathematics, 
# back to the original neuron to adjust their weights and biases in such a way that the loss is minimised.


import math

softmax_output = [0.7, 0.05, 0.25]            # suppose these are the final probabilities of some training sample
target_output1 = [1, 0, 0]
target_output2 = [0, 1, 0]

def categorical_cross_entropy_loss(target_output):
    return -1 * math.log(softmax_output[0]*target_output[0]
                        +softmax_output[1]*target_output[1]
                        +softmax_output[2]*target_output[2])

print(categorical_cross_entropy_loss(target_output1))
print(categorical_cross_entropy_loss(target_output2))


## See how the higher probability results in a lower loss (0.356675), while in case the target variables is the second one, which is probabilised to be with 
# only 5% confidence, then it brings in a loss of nearly 3!

