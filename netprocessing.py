import scipy, scipy.io
import numpy as np
import random
from weightclass import Theta

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
