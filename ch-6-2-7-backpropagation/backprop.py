import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = 0.5*(target-output)**2

# TODO: Calculate error term for output layer
output_error_term = (target-output) * (output) * (1-output)

# GK: a = b
# a = np.dot(output_error_term, weights_hidden_output)
# b = output_error_term * weights_hidden_output
# GK: c = d
# c = output_error_term * weights_hidden_output * hidden_layer_output * (1 - hidden_layer_output)
# d = np.dot(output_error_term,  weights_hidden_output * hidden_layer_output * (1 - hidden_layer_output))

# e = hidden_layer_output * (1 - hidden_layer_output)

# TODO: Calculate error term for hidden layer
hidden_error_term = np.dot(output_error_term, weights_hidden_output * hidden_layer_output * (1 - hidden_layer_output))

# TODO: Calculate change in weights for hidden layer to output layer
# GK result is: [0.00804047 0.00555918]
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
# GK: does not work
# delta_w_i_h = learnrate * hidden_error_term * x
# GK: transpose of a 1d row vector or a column vector w.r.t ndarray is the same array - no changes are made.
# delta_w_i_h = learnrate * hidden_error_term * x

# GK: update x to right shape
y = x[: ,None]
z = hidden_error_term[None, :]
delta_w_i_h = learnrate * np.dot( y, z )
# GK: This is the same as above
# delta_w_i_h = learnrate * hidden_error_term * x[:, None]


# GK: print output error term
# 0.028730669543515018
print('Output_error_term:')
print(output_error_term)


# GK: print hidden error term
# [ 0.00070802 -0.00204471]
print('hidden_error_term:')
print(hidden_error_term)


print('Change in weights for hidden layer to output layer:')
# [0.00804047 0.00555918]
print(delta_w_h_o)

print('Change in weights for input layer to hidden layer:')
# [[ 1.77005547e-04 -5.11178506e-04]
#  [ 3.54011093e-05 -1.02235701e-04]
#  [-7.08022187e-05  2.04471402e-04]]
print(delta_w_i_h)
