import scipy, scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from netprocessing import *
from evaluation import *
from dataprep import *
from weightclass import Theta

def run_nn():
    #test parameters to check against what my operations should actually result in.

    mat_contents = scipy.io.loadmat('sampleData.mat')
    x_mat = mat_contents['X']
    y_vec = mat_contents['y']
    y_mat = logical_y_matrix(y_vec,10)
    nn_specs = (400, 25, 10)
    lam = 1
    cost, trained_theta = train_nn(x_mat, y_mat, nn_specs, lam, max_iter = 100)
    return cost, trained_theta

def optimize_lambda():
    #I don't think that this works well
    mat_contents = scipy.io.loadmat('sampleData.mat')
    x_mat = mat_contents['X']
    y_vec = mat_contents['y']
    y_mat = logical_y_matrix(y_vec,10)
    nn_specs = (400, 25, 10)
    lam = 0.01
    #cost, train_theta, health, epsilon = train_nn(x_mat, y_mat, nn_specs, lam, max_iter = 100)
    #print('lambda:', lam,  '----- health status:', health)
    lam_cost_eps = []
    #lam_cost_eps.append([lam, cost, epsilon])
    upper_limit_found = False
    while not upper_limit_found:
        #this runs the training for 50 iterations with lambda 10x what it was previously
        average_cost = 0
        average_epsilon = 0
        stable = True
        for i in range(3):
            cost, train_theta, health, epsilon = train_nn(x_mat, y_mat, nn_specs, lam, max_iter = 100)
            if health == 'unstable':
                stable = False
            average_cost += cost
            average_epsilon += epsilon
        average_cost /= 3
        average_epsilon /= 3
        if not stable:
            print('reached unstable optimization! upper limit found')
            upper_limit_found = True
        elif stable:
            lam *= 3
        lam_cost_eps.append([lam, cost, epsilon])
        print('\n=====================\nlam:', lam, ' cost:', cost, ' eps:', epsilon)
        print('=====================\n\n')
        # cost, train_theta, health = train_nn(x_mat, y_mat, nn_specs, lam, max_iter = 100)
        # print('lambda:', lam,  '----- health status:', health)
        # lam_and_cost.append([lam, cost, health])

    print('lam_and_cost:\n', lam_cost_eps)
    return lam_cost_eps

def train_nn(X, y, layer_specs, lam, max_iter = 20 , eps_limit = 0.1, numerical_check = None, descent_type = 'batch', number_batches = 5, batch_order = 'ordered', batch_mod_type = 'end', plot_input = False):
    '''
    train the network's activation weights

    '''
    #initialize a random theta
    theta_train = Theta(layer_specs, 'random')

    if y.shape[0] != X.shape[0]:
        raise ValueError('X and y do not have the same number of training examples.')
    else:
        m = y.shape[0]


    eps = 1000 # default
    temp_cost = 100000000000000
    iteration_number = 0
    #feed forward, backpropagate and calculate cost
    #keep iterating until you pass max_iter or eps drops below eps_limit


    i = 0

    #setting up to plot
    x_plt = []
    y_plt = []
    plt.xlabel = '# Training Iterations'
    plt.ylabel = 'Cost (error function)'

    if plot_input == True:
        print('...live-plotting the first 200 training examples.')
    while (i <= max_iter):
        if descent_type == 'batch':
            h, a_values, z_values = feed_forward(theta_train, X, layer_specs)
            theta_train_grad = backprop(theta_train, y, h, a_values, z_values, lam, layer_specs)
            cost = calculate_cost(theta_train, y, h, lam)

        elif descent_type == 'mini-batch':
            #shuffle data here, as this marks the beginning of a new epoch
            X, y = shuffle_data(X, y)
            for j in range(number_batches):
                i_batch = i % number_batches
                x_mb, y_mb = select_slice(X, y, i_batch, number_batches)
                h, a_values, z_values = feed_forward(theta_train, x_mb, layer_specs)
                theta_train_grad = backprop(theta_train, y_mb, h, a_values, z_values, lam, layer_specs)
                cost = calculate_cost(theta_train, y_mb, h, lam)
                i += 1
            i -= 1 #makes up for an extra iteration count at the end of the while loop

        elif descent_type == 'mb':
            if batch_order == 'ordered':
                i_batch = i % number_batches
            elif batch_order == 'random':
                i_batch = random.randint(0, (number_batches - 1))

            #choose the slice of data you want, based on batch number

            x_mb, y_mb = select_slice(X, y, i_batch, number_batches)
            h, a_values, z_values = feed_forward(theta_train, x_mb, layer_specs)
            theta_train_grad = backprop(theta_train, y_mb, h, a_values, z_values, lam, layer_specs)
            cost  = calculate_cost(theta_train, y_mb, h, lam)


        elif descent_type == 'mini-batch-mod':
            if batch_order == 'ordered':
                i_batch = i % number_batches
            elif batch_order == 'random':
                i_batch = random.randint(0, (number_batches - 1))
            #FINISH THIS
            #this starts with mini-batch, and then changes to batch for the last few iterations
            if batch_mod_type == 'end':
                if i < (max_iter - int(max_iter / 20)):
                    x_mb, y_mb = select_slice(X, y, i_batch, number_batches)
                    h, a_values, z_values = feed_forward(theta_train, x_mb, layer_specs)
                    theta_train_grad = backprop(theta_train, y_mb, h, a_values, z_values, lam, layer_specs)
                    cost = calculate_cost(theta_train, y_mb, h, lam)
                elif i >= (max_iter - int(max_iter / 20)):
                    # perform full gradient descent for the last 1/20th of training iterations, to reach stability
                    h, a_values, z_values = feed_forward(theta_train, X, layer_specs)
                    theta_train_grad = backprop(theta_train, y, h, a_values, z_values, lam, layer_specs)
                    cost = calculate_cost(theta_train, y, h, lam)


            elif batch_mod_type == 'mixed':
                #this isn't very effective
                if i % int(m / 10) != 0:
                    #this catches almost all of the iterations
                    x_mb, y_mb = select_slice(X, y, i_batch, number_batches)
                    h, a_values, z_values = feed_forward(theta_train, x_mb, layer_specs)
                    theta_train_grad = backprop(theta_train, y_mb, h, a_values, z_values, lam, layer_specs)
                    cost = calculate_cost(theta_train, y_mb, h, lam)

                elif i % int(m / 10) == 0:
                    #this segment of normal gradient descent will happen 10 times during the training examples
                    for j in range(number_batches * 5):
                        h, a_values, z_values = feed_forward(theta_train, X, layer_specs)
                        theta_train_grad = backprop(theta_train, y, h, a_values, z_values, lam, layer_specs)
                        cost = calculate_cost(theta_train, y, h, lam)
                        i += 1 #still iterates it one time through for each iteration while it's in here
                    i -= 1 #to remove one iteration

        if (numerical_check == 'check_gradient' and i == 1):
            #only checks the gradient on the first iteration
            num_check = []
            if descent_type == 'batch':
                x_check = X
                y_check = y
            elif descent_type == 'mini-batch':
                x_check = x_mb
                y_check = y_mb

            num_grad = numerical_gradient(theta_train, x_check, y_check, layer_specs, lam)
            sum_difference = 0
            m_theta = theta_train_grad.get_flat().shape[0]

            for j in range(num_grad.get_flat().shape[0]):
                difference = (theta_train_grad.row[j] - num_grad.row[j])
                num_check.append([theta_train_grad.row[j], num_grad.row()[j], difference])

            print('backprop gradient - numerical gradient - difference')
            for k in range(0, num_grad.row.shape[0], int(num_grad.row.shape[0]/20)):
                print(num_check[k])
            # average_difference = sum_difference / theta_train_grad.get_flat().shape[0]
            # print('average difference between numerical gradient and backprop gradient:\n', average_difference)
            num_check = np.array(num_check)
            #print('backprop gradient - numerical gradient - difference\n', num_check)
            input('press enter to continue training neural network weights.')

        #perform gradient descent, alter theta_train
        theta_train = grad_descent(theta_train, theta_train_grad, 2.0)

        #print('Theta after grad_descent:', theta_train.get_flat())
        #update while loop conditional values
        eps = (temp_cost - cost)
        print('cost after iteration '+ str(i) + ': ' + str(cost), end='\r')

        i += 1
        #catches unstable optimization for batch gradient descent
        if (descent_type == 'batch' and eps < 0):
            print('\n------------------\nUnstable optimization, training stopped.')
            print('previous cost:', temp_cost)
            print('current cost:', cost)
            print('------------------\n\n')
            health_status = 'unstable'
            return cost, theta_train, health_status, eps

        temp_cost = cost #set temp_cost for next iteration
        if plot_input == True:

            x_plt.append(i)
            y_plt.append(cost)
            # if i % 10 == 0:
            #live plotting, prints for the first 200 iterations of training
            if i < 200:
                plt.plot(x_plt, y_plt)
                plt.pause(0.05)
            if i > (max_iter - 30):
                #live plotting for the last 30 iterations
                plt.plot(x_plt, y_plt)
                plt.pause(0.05)



    plt.close()
    if plot_input == True:
        print('\n...plotting entire training cost')
        plt.plot(x_plt, y_plt)
        plt.show()


    print('training of weights is now complete.                               ')
    health_status = 'stable'
    print('final cost:', cost)
    print('change in cost from previous iteration:', eps)
    return cost, theta_train, health_status, eps

x_mat, y_mat = load_data()
x_all_shuffled, y_all_shuffled = shuffle_data(x_mat, y_mat)

x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(x_all_shuffled, y_all_shuffled, 0.9, 0.0, 0.1)

neural_net_specs = (400, 25, 10)

print('training neural network...')
trained_cost, trained_theta, trained_health, trained_eps = train_nn(x_train, y_train, neural_net_specs, lam = 0.05, max_iter = 1000, eps_limit = 0.1, numerical_check = None, descent_type = 'mini-batch', number_batches = 7, batch_order = 'ordered', batch_mod_type = 'end', plot_input = True)

evaluate_nn(trained_theta, x_test, y_test, neural_net_specs)
