import scipy, scipy.io
import numpy as np
from netprocessing import *
from dataprep import *
from weightclass import Theta

def predict_values(trained_theta, testing_x, nn_specs, cutoff = 0.5, print_check = False):
    #cutoff is the point at which the program counts a number either as a 1 or 0
    m = testing_x.shape[0]
    h = feed_forward(trained_theta, testing_x, nn_specs)[0]
    output_values = []
    for i in range(m):
        #go through all the rows
        max_index = np.argmax(h[i,:])
        if h[i, max_index] >= cutoff:
            output_values.append([max_index])
        elif h[i, max_index] < cutoff:
            output_values.append([max_index])

        if ((i % 100 == 0) and (print_check == True)): #grabs every 100th example to check
            print('output number =', max_index, '\n here is the row:\n', h[i,:])
    output_values = np.array(output_values)
    return output_values

def evaluate_nn(trained_theta, x_test, y_test, net_specs):
    #reformat y_test from 0 0 0 0 1 0 0 0 0  to 4
    y_correct = []
    for i in range(y_test.shape[0]):
        max_index = np.argmax(y_test[i,:])
        y_correct.append([max_index])
    y_correct = np.array(y_correct)

    #test output with non-shuffled values
    output_vals = predict_values(trained_theta, x_test, net_specs)

    number_wrong = 0
    for i in range(y_test.shape[0]):

        if y_correct[i,0] != output_vals[i,0]:
            number_wrong += 1

    m = y_correct.shape[0]
    print('total number incorrectly labeled:', number_wrong)
    print('total test examples:', m)
    print('percent correct:', round((((m - number_wrong) / m) * 100), 4))

def test_vals():
    theta_contents = scipy.io.loadmat('ex4weights.mat')
    theta1 = theta_contents['Theta1']
    theta2 = theta_contents['Theta2']
    theta2_flat = theta2.flatten()
    theta1_flat = theta1.flatten()
    th = np.concatenate((theta1_flat, theta2_flat))
    test_theta = Theta(nn_specs, row_entry = th)

    h_val, a_values, z_values = feed_forward(test_theta, x_mat, nn_specs)
    theta_train_grad = backprop(test_theta, y_mat, h_val, a_values, z_values, lam, nn_specs)
    cost = calculate_cost(test_theta, y_mat, h_val, lam)
    print('cost at fixed debugging parameters w/ lambda =', lam, ':', cost)
