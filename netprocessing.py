import scipy, scipy.io
import numpy as np
import random
from weightclass import Theta
from dataprep import *

def sigmoid(z):
    sig = (1.0 /(1.0+np.exp(-1.0 * z)))
    return sig

def sigmoid_gradient(z):
    sigGrad = sigmoid(z)*(1-sigmoid(z))
    return sigGrad

def calculate_cost(theta, y, h, lam):
    m = y.shape[0]
    #CALCULATE COST BASED ON CURRENT THETA
    #====================================================
    jMatrix = ((-y * np.log(h))-((1-y)*np.log(1-h)))
    jTheta = (1/m)*np.sum(jMatrix)

    theta_sums = 0
    for theta_i in range(len(theta.net_specs) - 1):
        theta_sums += np.sum(np.square(theta.get_matrix(theta_i)[:,1:]))
    jTheta += (lam/(2*m))*theta_sums

    return jTheta

def grad_descent(theta, gradients, learning_rate):
    '''
    Takes one step in the gradient descent direction
    '''
    theta.row = theta.row - learning_rate * gradients.row
    return theta

def numerical_gradient_V2(theta, X, y, lam):
    #WORK ON THIS!!! WORK ON THIS !!! WORK ON THIS !!!
    offset = 0.0001
    #set theta to theta flat
    #set an array of zeros, iterate through adding the offset, then run the numerical gradient
    #iterate through and add the offset, run the
    delta = Theta(theta.net_specs, 'zeros')

    num_grad = Theta(theta.net_specs, 'zeros')

    for i in range(len(delta.row)):
        print('...checking gradient number: '+str(i+1)+'/'+str(len(delta.row)), end='\r')
        # print('......running through each gradient.......', i, '/', delta.size)
        delta.row[i] = offset
        new_theta = theta - delta
        h, a_values, z_values = feed_forward_V2(new_theta, X)
        loss1 = calculate_cost(new_theta, y, h, lam)

        new_theta = theta + delta
        #print('new_theta addition:', new_theta, '\n=====================')
        h, a_values, z_values = feed_forward_V2(new_theta, X)
        loss2 = calculate_cost(new_theta, y, h, lam)

        #compute gradient here
        num_grad.row[i] = (loss2 - loss1) / (2 * offset)
        delta.row[i] = 0 #return to zeros all around
    return num_grad

def feed_forward_V2(theta, X):
    #THIS WORKS!!!!!!!!
    m = X.shape[0]
    a_dict = {}
    z_dict = {}
    for layer in range(len(theta.net_specs)):
        if layer == 0:
            #first layer
            ones = np.ones((X.shape[0],1))
            a_dict[layer] = np.concatenate((ones, X), axis = 1)
        elif 0 < layer < (len(theta.net_specs)-1):
            #middle layers
            z_dict[layer] = a_dict[layer-1] @ theta.get_matrix(layer-1).transpose()
            a_val = sigmoid(z_dict[layer])
            ones = np.ones((z_dict[layer].shape[0],1))
            a_dict[layer] = np.concatenate((ones, a_val), axis = 1)
        elif layer == (len(theta.net_specs)-1):
            #last layer
            z_dict[layer] = a_dict[layer-1] @ theta.get_matrix(layer-1).transpose()
            a_dict[layer] = sigmoid(z_dict[layer])
            h = a_dict[layer]
    return h, a_dict, z_dict

def backprop_V2(theta, y, h, a_dict, z_dict, lam):
    #THIS IS THE ONE
    backprop_grad = Theta(theta.net_specs, 'zeros')
    m = y.shape[0]
    del_dict = {}
    theta_grad = Theta(theta.net_specs,'zeros')
    for layer in range((len(theta.net_specs)-1), 0, -1):
        if layer == (len(theta.net_specs) - 1):
            del_dict[layer] = h - y
        else:
            del_dict[layer] = (del_dict[layer+1] @ theta.get_matrix(layer))[:,1:] * sigmoid_gradient(z_dict[layer])
        theta_grad = (1/m) * (del_dict[layer].transpose() @ a_dict[layer-1])
        theta_grad[:, 1:] += (lam/m) * theta.get_matrix(layer-1)[:,1:]
        backprop_grad.set_matrix((layer-1),theta_grad)

    return backprop_grad

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
