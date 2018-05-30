import scipy, scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from math import exp
from netprocessing import *
from evaluation import *
from dataprep import *
from weightclass import Theta
from numpy.linalg import norm

def check_gradient(X, y, net_specs, lam):
    x_shuffle, y_shuffle = shuffle_data(X, y)
    x_check, y_check = select_slice(x_shuffle, y_shuffle, batch_number = random.randint(0,9), number_batches = 10)
    theta_check = Theta(net_specs, initialization = 'random')
    #numerical gradient
    numer_grad = numerical_gradient_V2(theta_check, x_check, y_check, lam)
    #backprop gradient
    h, a_values, z_values = feed_forward_V2(theta_check, x_check)
    training_grad = backprop_V2(theta_check, y_check, h, a_values, z_values, lam)
    error_grad = numer_grad - training_grad
    normalized_error = norm(training_grad.row - numer_grad.row) / norm(training_grad.row + numer_grad.row)
    print('normalized error with numerical gradient checking:', normalized_error)

    if abs(normalized_error) <= 0.000000001:
        correct_gradient = True
    elif abs(normalized_error) > 0.000000001:
        correct_gradient = False

    return correct_gradient

def cost_func(theta, X, y, lam):
    h = feed_forward_V2(theta, X)[0]
    cost = calculate_cost(theta, y, h, lam)
    return cost

def grad_func(theta, X, y, lam):
    h, a_values, z_values = feed_forward_V2(theta, X)
    grad = backprop_V2(theta, y, h, a_values, z_values, lam)
    return grad

def full_batch(X, y, theta, lam, max_iter, alpha, iter_per_epoch = 10, store_values = False):
    total_iter = 0
    epoch = 0
    #standard is to shuffle every ten iterations, can be specified otherwise
    number_epochs = max_iter // iter_per_epoch
    iteration_store = []
    cost_store = []
    while epoch < number_epochs:
        x_shuffle, y_shuffle = shuffle_data(X, y)
        # if ((epoch+1) % 20 == 0) and (epoch != 0):
        #     alpha = alpha * 0.9
        for i in range(iter_per_epoch):
            #cost, training_grad = cost_and_grad(theta, X, y, lam)
            cost = cost_func(theta, X, y, lam)
            training_grad = grad_func(theta, X, y, lam)
            theta = grad_descent(theta, training_grad, learning_rate = alpha)
            total_iter += 1
            print('epoch: '+ str(epoch+1) + ' cost: ' + str(cost), end = '\r')
            if store_values:
                iteration_store.append(total_iter)
                cost_store.append(cost)
        epoch += 1
    print()
    return theta, cost, iteration_store, cost_store

def mini_batch(X, y, theta, lam, number_epochs, batch_divisions, alpha, store_values = False):
    total_iter = 0
    epoch = 0
    iteration_store = []
    cost_store = []
    while epoch < number_epochs:
        x_shuffle, y_shuffle = shuffle_data(X, y)
        # if ((epoch+1) % 20 == 0) and (epoch != 0):
        #     alpha = alpha * 0.9

        for i in range(batch_divisions):
            x_mb, y_mb = select_slice(x_shuffle, y_shuffle, batch_number = i, number_batches = batch_divisions)
            cost = cost_func(theta, x_mb, y_mb, lam)
            training_grad = grad_func(theta, x_mb, y_mb, lam)
            theta = grad_descent(theta, training_grad, learning_rate = alpha)
            total_iter += 1
            print('alpha: ' + str(round(alpha, 3))+ ' epoch: '+ str(epoch+1) + ' cost: ' + str(cost), end = '\r')
            if store_values:
                iteration_store.append(total_iter)
                cost_store.append(cost)
        epoch += 1
    print()
    return theta, cost, iteration_store, cost_store

def train_net(X, y, net_specs, batch_method = 'full-batch', max_iter = 1000, batch_divisions = 5, lam = 0.05, alpha = 2.0, gradient_check = False, plotter = True):
    training_theta = Theta(net_specs, 'random')
    if y.shape[0] != X.shape[0]:
        raise ValueError('X and y do not have the same number of training examples.')
    else:
        m = y.shape[0]

    if gradient_check == True:
        backprop_correct = check_gradient(X, y, net_specs, lam)
        if backprop_correct:
            print('backpropagation is functioning correctly.')
        elif not backprop_correct:
            print('WARNING! backpropagation is not functioning properly.')


    if batch_method == 'mini-batch':
        num_epochs = max_iter // batch_divisions
        training_theta, cost, iteration_store, cost_store = mini_batch(X, y, training_theta, lam, num_epochs, batch_divisions, alpha =  alpha, store_values = plotter)
    elif batch_method == 'full':
        training_theta, cost, iteration_store, cost_store = full_batch(X, y, training_theta, lam, max_iter, alpha = alpha, store_values = plotter)

    if plotter:
        x_plt = iteration_store
        y_plt = cost_store
        plt.plot(x_plt, y_plt)
        plt.show()
    return training_theta, cost

x_mat, y_mat = load_data()
x_all_shuffled, y_all_shuffled = shuffle_data(x_mat, y_mat)

x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(x_all_shuffled, y_all_shuffled, 1.0, 0.0, 0.0)

lam_train = 0.03
neural_net_specs = (400, 25, 10)

final_theta, final_cost = train_net(x_train, y_train, neural_net_specs, batch_method = 'mini-batch', max_iter = 1000, batch_divisions = 5, lam = lam_train, gradient_check = False)


# sample_theta = load_sample_theta(neural_net_specs)
# h, a_dict, z_dict = feed_forward_V3(sample_theta, x_mat, neural_net_specs)
# theta_grad = backprop_V5(sample_theta, y_mat, h, a_dict, z_dict, test_lam, neural_net_specs)
# numerical_grad = numerical_gradient_V2(sample_theta, x_mat, y_mat, neural_net_specs, test_lam)
#
# error_grad = theta_grad - numerical_grad
# np.savetxt("matrix0.csv", error_grad.get_matrix(0), delimiter=",")
# np.savetxt("matrix1.csv", error_grad.get_matrix(1), delimiter=",")
