import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

expected_op = [0.01,0.99]
input = [0.05,0.1]
#weights_l1 = np.random.uniform(0,1,size=(2,2))
#weights_l2 = np.random.uniform(0,1,size=(2,2))
weights_l1 = [[0.15,0.2],[0.25,0.3]]
weights_l2 = [[0.4,0.45],[0.5,0.55]]
a = 1

for i in range(0,10000):
    # Input to hidden layer
    hidden_node = np.dot(input,weights_l1)      # Matrix (1*9).(9*5) = (1*5) hidden layer
    hidden_node = [sigmoid(x) for x in hidden_node]
    #self.hidden_layer[i][j] = hidden_node

    # Hidden to output layer
    output_node = np.dot(hidden_node,weights_l2)       # Matrix (1*5).(5*3) = (1*3) output layer
    output_node = [sigmoid(x) for x in output_node]
    #self.output_layer[i][j] = output_node

    #print("Initial output")
    #print(output_node)

    sqr_error = 0
    for k in range(2):
        sqr_error += (0.5* (expected_op[k] - output_node[k])**2)

    # Backpropogate error
    # Ouput to hidden layer error back propogation
    for h in range(2):
        for k in range(2):
            delta_output = output_node[k] * (1 - output_node[k]) * \
                (output_node[k] - expected_op[k]) * hidden_node[h]
            # Update hidden layer weights
            weights_l2[h][k] -= (a * delta_output)


    # Hidden layer to input layer error back propogation
    for ip in range(2):
        for h in range(2):
            sum_error = 0
            for k in range(2):
                sum_error += (output_node[k] * (1 - output_node[k]) * \
                    (output_node[k] - expected_op[k]) * weights_l2[h][k])

            # sum error is the error in the hidden layer for each hidden node.
            delta_output = sum_error * hidden_node[h] * (1 - hidden_node[h]) * input[ip]
            weights_l1[ip][h] -= (a * delta_output)

    #print("Weights Layer1")
    #print(weights_l1)
    #print("Weights Layer2")
    #print(weights_l2)

    # Feed forward again to verify
    # Input to hidden layer
    hidden_node = np.dot(input,weights_l1)      # Matrix (1*9).(9*5) = (1*5) hidden layer
    hidden_node = [sigmoid(x) for x in hidden_node]
    #self.hidden_layer[i][j] = hidden_node

    # Hidden to output layer
    output_node = np.dot(hidden_node,weights_l2)       # Matrix (1*5).(5*3) = (1*3) output layer
    output_node = [sigmoid(x) for x in output_node]

    print("Final output")
    print(output_node)

    sqr_error = 0
    for k in range(2):
        sqr_error += (0.5* (expected_op[k] - output_node[k])**2)

    print("Squared error")
    print(sqr_error)

