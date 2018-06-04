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

def main():
    x_mat, y_mat = load_data()
    x_all_shuffled, y_all_shuffled = shuffle_data(x_mat, y_mat)

    x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(x_all_shuffled, y_all_shuffled, 0.9, 0.0, 0.1)

    lam_train = 0.03
    neural_net_specs = (400, 25, 10)

    final_theta, final_cost = train_net(x_train, y_train, neural_net_specs, batch_method = 'mini-batch', max_iter = 500, batch_divisions = 5, lam = lam_train, gradient_check = False)
    percentage_correct = evaluate_nn(final_theta, x_test, y_test, neural_net_specs)

    write_results_to_csv(percentage_correct, final_cost, final_theta)

    return final_theta, final_cost

if __name__ == '__main__':
    main()
