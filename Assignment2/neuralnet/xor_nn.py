import numpy as np
import sigmoid as sg

epochs = 1
input_size, hidden_size, output_size = 2, 3, 1
LR = .1 # learning rate

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([ [0],   [1],   [1],   [0]])

w_hidden = np.random.uniform(size=(input_size, hidden_size))
w_output = np.random.uniform(size=(hidden_size, output_size))

for epoch in range(epochs):
 
    # Forward
    act_hidden = sg.sigmoid(np.dot(X, w_hidden))
    print act_hidden
    output = np.dot(act_hidden, w_output)
    print output

    # Calculate error
    error = y - output

    print error

    if epoch % 5000 == 0:
        sumerr = sum(error)
        print "Sum of errors",
        print sumerr

    # Backward
    dZ = error * LR
    w_output += act_hidden.T.dot(dZ)
    dH = dZ.dot(w_output.T) * sg.sigmoid_prime(act_hidden)
    w_hidden += X.T.dot(dH)
