import scipy, scipy.io
import numpy as np
import random
import pandas as pd
from weightclass import WeightMatrices
import reference_functions as rf
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class FeedForwardNN(object):
    def __init__(self, nn_specs, lam = 0.05, learning_rate = 0.5, transfer_func_name = 'sigmoid', training_method = 'mini-batch'):
        self.training_method = training_method
        self.transfer_func_name = transfer_func_name
        self.specs = nn_specs
        self.theta = FeedForwardNN.initialize_random_weights(self)
        self.lam = lam
        self.learning_rate = learning_rate

    def initialize_random_weights(self):
        return WeightMatrices(self.specs, initialization = 'random')

    def load_data(self, file_name, file_type = 'mat', split_ratios = (0.6, 0.2, 0.2)):
        if file_type == 'mat':
            mat_contents = scipy.io.loadmat(file_name)
            self.x_all = mat_contents['X']
            self.y_all = rf.logical_y_matrix(mat_contents['y'],self.specs[-1])
        else:
            raise ValueError('Not compatible with that input data type')
        self.x_all, self.y_all = rf.shuffle_data(self.x_all, self.y_all)
        self.x_train, self.y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = rf.split_data(self.x_all, self.y_all, split_ratios)

    @staticmethod
    def normalize_x(xdata):
        new_x_data = np.array((xdata - xdata.mean(axis = 0)) / (xdata.max(axis = 0) - xdata.min(axis = 0)), dtype = float)
        return new_x_data
    @staticmethod
    def categorize_y(ydata):
        new_y_data = np.array(pd.get_dummies(ydata).values, dtype = float)
        return new_y_data

    def set_data(self, xdata, ydata, split_ratios = (0.6, 0.2, 0.2)):
        #manually set data (formats that require more reading)
        x = FeedForwardNN.normalize_x(xdata)
        y = FeedForwardNN.categorize_y(ydata)
        self.x_all, self.y_all = rf.shuffle_data(x, y)
        self.x_train, self.y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = rf.split_data(self.x_all, self.y_all, split_ratios)

    def set_train_data(self, xdata, ydata):
        self.x_train = xdata
        self.y_train = ydata

    def set_vali_data(self, xdata, ydata):
        self.x_vali = xdata
        self.y_vali = ydata

    def set_test_data(self, xdata, ydata):
        self.x_test = xdata
        self.y_test = ydata

    def transfer_func(self, z):
        if self.transfer_func_name == 'sigmoid':
            transfer = (1.0 /(1.0+np.exp(-1.0 * z)))
        return transfer

    def transfer_derivative(self, z):
        if self.transfer_func_name == 'sigmoid':
            derivative = FeedForwardNN.transfer_func(self, z)*(1-FeedForwardNN.transfer_func(self, z))
        return derivative

    def feedforward(self, theta, x_ff):
        m = x_ff.shape[0]
        a_dict = {}
        z_dict = {}
        for layer in range(len(self.specs)):
            if layer == 0:
                #first layer
                ones = np.ones((x_ff.shape[0],1))
                a_dict[layer] = np.concatenate((ones, x_ff), axis = 1)
            elif 0 < layer < (len(self.specs)-1):
                #middle layers
                z_dict[layer] = a_dict[layer-1] @ theta.get_matrix(layer-1).transpose()
                a_val = FeedForwardNN.transfer_func(self, z_dict[layer])
                ones = np.ones((z_dict[layer].shape[0],1))
                a_dict[layer] = np.concatenate((ones, a_val), axis = 1)
            elif layer == (len(self.specs)-1):
                #last layer
                z_dict[layer] = a_dict[layer-1] @ theta.get_matrix(layer-1).transpose()
                a_dict[layer] = FeedForwardNN.transfer_func(self, z_dict[layer])
                h = a_dict[layer]
        self.h = h
        self.a_dict = a_dict
        self.z_dict = z_dict
        return h

    def backprop(self, theta, y_bp, h_bp):
        backprop_grad = WeightMatrices(self.specs, 'zeros')
        m = y_bp.shape[0]
        del_dict = {}
        for layer in range((len(self.specs)-1), 0, -1):
            if layer == (len(self.specs) - 1):
                del_dict[layer] = h_bp - y_bp
            else:
                del_dict[layer] = (del_dict[layer+1] @ theta.get_matrix(layer))[:,1:] * FeedForwardNN.transfer_derivative(self, self.z_dict[layer])
            theta_grad = (1/m) * (del_dict[layer].transpose() @ self.a_dict[layer-1])
            theta_grad[:, 1:] += (self.lam/m) * theta.get_matrix(layer-1)[:,1:]
            backprop_grad.set_matrix((layer-1),theta_grad)
        return backprop_grad

    def calculate_cost(self, theta, y_cc, h_cc):
        m = y_cc.shape[0]
        jMatrix = ((-y_cc * np.log(h_cc))-((1-y_cc)*np.log(1-h_cc)))
        jTheta = (1/m)*np.sum(jMatrix)
        theta_sums = 0
        for theta_i in range(len(self.specs) - 1):
            theta_sums += np.sum(np.square(theta.get_matrix(theta_i)[:,1:]))
        jTheta += (self.lam/(2*m))*theta_sums
        return jTheta

    def grad_descent(self, theta, theta_grad):
        theta.row = theta.row - self.learning_rate * theta_grad.row
        return theta

    def train_cycle(self, x, y):
        h = FeedForwardNN.feedforward(self, self.theta, x)
        self.backprop_grad = FeedForwardNN.backprop(self, self.theta, y, h)
        self.theta = FeedForwardNN.grad_descent(self, self.theta, self.backprop_grad)
        return FeedForwardNN.calculate_cost(self, self.theta, y, h)

    def get_current_cost(self, data_select = 'train'):
        if data_select == 'train':
            y = self.y_train
            x = self.x_train
        elif data_select == 'validate':
            y = self.y_vali
            x = self.x_vali
        elif data_select == 'test':
            y = self.y_test
            x = self.x_test
        h = FeedForwardNN.feedforward(self, self.theta, x)
        return FeedForwardNN.calculate_cost(self, self.theta, y, h)

    def numerical_gradient(self, theta, x_ng, y_ng):
        offset = 0.0001
        delta = WeightMatrices(self.specs, 'zeros')
        num_grad = WeightMatrices(self.specs, 'zeros')
        for i in range(len(delta.row)):
            print('...checking gradient number: '+str(i+1)+'/'+str(len(delta.row)), end='\r')
            delta.row[i] = offset
            h = FeedForwardNN.feedforward(self, (theta - delta), x_ng)
            loss1 = FeedForwardNN.calculate_cost(self, (theta - delta), y_ng, h)
            h = FeedForwardNN.feedforward(self, (theta + delta), x_ng)
            loss2 = FeedForwardNN.calculate_cost(self, (theta + delta), y_ng, h)
            num_grad.row[i] = (loss2 - loss1) / (2 * offset)
            delta.row[i] = 0 #return to zeros all around
        return num_grad

    def verify_gradient(self):
        x_check, y_check = rf.select_slice(self.x_all, self.y_all, number_batches = 10)
        theta_check = WeightMatrices(self.specs, 'random')
        num_grad = FeedForwardNN.numerical_gradient(self, theta_check, x_check, y_check)
        hypothesis = FeedForwardNN.feedforward(self, theta_check, x_check)
        backprop_grad = FeedForwardNN.backprop(self, theta_check, y_check, hypothesis)
        error = np.linalg.norm(backprop_grad.row - num_grad.row)/np.linalg.norm(backprop_grad.row + num_grad.row)
        print('backprop gradient values:\n', backprop_grad.row)
        print('numerical gradient values:\n', num_grad.row)
        print('normalized error with numerical gradient checking:', error)
        if abs(error) <= 0.000000001:
            return True
        elif abs(error) > 0.000000001:
            return False

    def mini_batch(self, x_train, y_train, num_batches):
        x_ts, y_ts = rf.unison_shuffle(x_train, y_train)
        for batch_i in range(num_batches):
            x_batch, y_batch = rf.select_slice(x_ts, y_ts, batch_i, num_batches)
            cost = FeedForwardNN.train_cycle(self, x_batch, y_batch)

    def full_batch(self, x_train, y_train):
        x_ts, y_ts = rf.unison_shuffle(x_train, y_train)
        cost = FeedForwardNN.train_cycle(self, x_ts, y_ts)

    def mixed_batch(self, x_train, y_train, num_batches, num_full_iter):
        FeedForwardNN.mini_batch(self, x_train, y_train, num_batches)
        for i in range(num_full_iter):
            FeedForwardNN.full_batch(self, x_train, y_train)

    def train_network(self, max_epochs = 10000000, number_batches = 5):
        time_start = time.time()
        epoch_list = []
        vali_cost_list = []
        train_cost_list = []
        print('\n================================\ntraining network with', self.training_method, 'training method...')
        i = 0
        validation_error_rising = False
        while (validation_error_rising == False and i < max_epochs):
            if self.training_method == 'mini-batch':
                FeedForwardNN.mini_batch(self, self.x_train, self.y_train, number_batches)
            if self.training_method == 'full-batch':
                FeedForwardNN.full_batch(self, self.x_train, self.y_train)
            if self.training_method == 'mixed-batch':
                number_full = number_batches // 2
                FeedForwardNN.mixed_batch(self, self.x_train, self.y_train, number_batches, number_full)

            train_cost = FeedForwardNN.get_current_cost(self, 'train')
            vali_cost = FeedForwardNN.get_current_cost(self, 'validate')

            epoch_list.append(i)
            vali_cost_list.append(vali_cost)
            train_cost_list.append(train_cost)
            print(' epoch: '+str(i)+' training cost: '+str(round(train_cost,3))+' validation cost: '+str(round(vali_cost,3)), end = '\r')
            i += 1
            validation_error_rising = rf.check_test_vali_error(vali_cost_list, lookback = 40)
        time_end = time.time()
        print('time to train:', round(time_end - time_start,2), 'seconds')
        print('epoch: '+str(i)+' training cost: '+str(round(train_cost,3))+' validation cost: '+str(round(vali_cost,3)))
        min_vali_cost_iter = np.argmin(vali_cost_list)
        print(min_vali_cost_iter, 'iterations produced the lowest validation cost:', vali_cost_list[min_vali_cost_iter])
        self.epoch_list = epoch_list
        self.vali_cost_list = vali_cost_list
        self.train_cost_list = train_cost_list

    def check_nn_accuracy(self):
        rf.evaluate_nn(self, self.x_test, self.y_test)

    def plot_training_curves(self, close_timer = False):
        plt.title('Training Error vs Validation Error')
        plt.plot(self.epoch_list, self.train_cost_list)
        plt.plot(self.epoch_list, self.vali_cost_list)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Error')
        plt.legend(['training error', 'validation error'])
        if close_timer:
            print('...plotting for', close_timer, 'seconds...\n')
            plt.pause(close_timer)
        else:
            plt.show()
