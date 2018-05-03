import scipy, scipy.io
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class Theta(object):
    #requires inputs of neural net specs (2,4,3)--number of nodes in each layer
    def __init__(self, neural_net_specs, code = 0):
        #theta set to 0 as default
        self.specs = neural_net_specs
        #initialize random theta
        self.matrices = {}
        if code == 0:
            #if no input to set theta, then it sets theta to zeros
            Theta.set_zeros(self)
        elif code == 1:
            #generate random theta with code 1
            Theta.generate_random_theta(self)

        #flattens theta upon initialization
        Theta.get_flat(self)

    def get_flat(self):
        for i_theta in range(len(self.matrices)):
            if i_theta == 0:
                self.flat = self.matrices[i_theta].flatten()
            else:
                self.flat = np.concatenate((self.flat, self.matrices[i_theta].flatten()))
        return self.flat

    def generate_random_theta(self, eps = 0.3):
        #generates a dictionary of random theta values, with a dictionary key
        #correlating to it's position in the neural network
        #eps is the range in which the random values are generated (+/- eps)
        for i in range(len(self.specs)-1):
            theta_dims = (self.specs[i+1], self.specs[i]+1)
            random_theta = (np.random.rand(theta_dims[0],theta_dims[1]))*2*eps - eps
            self.matrices[i] = random_theta

    def set_theta(self, new_theta):

        if type(new_theta) == list:
            #convert to numpy array
            new_theta = np.array(new_theta)

        if new_theta.size == new_theta.shape[0]:
            #catches if it is a row vector

            #counts the number of elements required, checks against input
            count = 0
            for i in range(len(self.specs)-1):
                dims = Theta.required_theta_dims(self, i)
                count += (dims[0] * dims [1])
            if new_theta.size != count:
                raise TypeError('Existing thetas and new theta have different number of values.')

            #writes the input theta to the matrix structure
            temp_index = 0
            for i in range(len(self.specs)-1):
                theta_shape = Theta.required_theta_dims(self, i)
                num_values = theta_shape[0] * theta_shape[1]
                self.matrices[i] = np.reshape(new_theta[temp_index:(temp_index+num_values)],theta_shape)
                temp_index += num_values
        else:
            #throw error, maybe write some code later to handle matrix theta inputs
            for i in range(0, (len(self.specs)-1)):
                required_dims = Theta.required_theta_dims(self, i) #generates the acceptable size
                #checks if there is an existing dictionary with the same key
                if new_theta.shape == required_dims: #new theta is same shape as expected shape
                    self.matrices[i] = new_theta
                elif (required_dims[0] * required_dims[1]) == new_theta.size: #row vector
                    self.matrices[i] = np.reshape(new_theta, required_dims)
                else:
                    raise TypeError('Incorrect size or shape for theta, based on the neural net specs.')
        #flattens theta again
        Theta.get_flat(self)
        return self.matrices

    def set_theta_ind(self, new_th, new_key):
        if type(new_theta) == list:
            #convert to numpy array
            new_th = np.array(new_th)

        required_dims = Theta.required_theta_dims(self, new_key)
        if new_theta.shape == required_dims: #new theta is same shape as expected shape
            self.matrices[new_key] = new_th
        elif (required_dims[0] * required_dims[1]) == new_th.size: #row vector
            self.matrices[new_key] = np.reshape(new_th, required_dims)
        else:
            raise TypeError('Incorrect size or shape for theta, based on the neural net specs.')

    def required_theta_dims(self, dict_key):
        dict_key = int(dict_key) #forces it into an integer
        if dict_key > (len(self.specs) - 1): #catches theta outside of the acceptable range
            raise TypeError('You entered a theta value that is outside of the limits of this neural net.')
        theta_dims = (self.specs[dict_key + 1], (self.specs[dict_key]+1))
        return theta_dims

    def set_zeros(self):
        for i in range(len(self.specs)-1):
            self.matrices[i] = np.zeros((self.specs[i+1], (self.specs[i]+1)))
        Theta.get_flat(self)
        #return self.matrices

    def get_theta_matrices(self):
        return self.matrices

    def __getitem__(self, key):

        return self.matrices[key]

    def __setitem__(self, key, item):

        self.matrices[key] = item

    def __str__(self):
        return str(self.matrices)

def sigmoid(z):
    sig = (1.0 /(1.0+np.exp(-1.0 * z)))
    return sig

def sigmoid_gradient(z):
    sigGrad = sigmoid(z)*(1-sigmoid(z))
    return sigGrad

def logical_y_matrix(vector_y, number_labels):
    '''
    creates a logical array for classification:
    input: [[2],[3],[1]]
    output: [[0,1,0],[0,0,1],[1,0,0]]
    '''
    if type(vector_y) == list:
        vector_y = np.array(vector_y)

    logical_y = np.zeros((vector_y.shape[0],number_labels))
    for i in range(logical_y.shape[0]):
        logical_y[i,(int(vector_y[i])-1)] = 1.0

    return logical_y

def feed_forward(theta, X, layer_specs):
    #REMOVE y from this and from all instances where this function is called
    m = X.shape[0]

    a_dict = {} #dict of the nodes
    z_dict = {} #dict of the intermediate values
    #FEED FORWARD
    for layer in range(len(layer_specs)):
        if layer == 0:
            #first layer
            ones = (np.ones((np.shape(X)[0],1)))
            aVal = np.append(ones, X, axis = 1)
            a_dict[layer] = aVal
        elif layer == (len(layer_specs)-1):
            #last layer
            zVal = a_dict[(layer-1)] @ theta[(layer-1)].transpose()
            aVal = sigmoid(zVal)
            a_dict[layer] = aVal
            h = aVal

        else:
            #middle layers
            zVal = a_dict[layer-1] @ theta[layer-1].transpose()
            z_dict[layer] = zVal
            #calculate the sigmoid of the zValue
            aVal = sigmoid(zVal)
            #add a column of ones since it's a middle layer, accounting for bias unit
            ones = (np.ones((np.shape(aVal)[0],1)))
            aVal = np.append(ones, aVal, axis = 1)
            a_dict[layer] = aVal
    #====================================================
    #need to return hypothesis (h), a_dict, z_dict
    return h, a_dict, z_dict

def calculate_cost(theta, y, h, lam):
    m = y.shape[0]
    #CALCULATE COST BASED ON CURRENT THETA
    #====================================================
    jMatrix = ((-y * np.log(h))-((1-y)*np.log(1-h)))
    jTheta = (1/m)*np.sum(jMatrix)

    theta_sums = 0
    for theta_i in range(len(theta.matrices)):
        theta_sums += np.sum(np.square(theta[theta_i][:,1:]))
    jTheta += (lam/(2*m))*theta_sums

    return jTheta, jMatrix

def backprop(theta, y, h, a_dict, z_dict, lam, layer_specs):

    #initialize a gradient theta object
    #====================================================

    #THIS MIGHT NOT WORK
    m = y.shape[0]
    grad_theta = Theta(layer_specs)
    grad_theta.set_zeros()

    del_dict = {}
    for i in range((len(layer_specs)-1), 0, -1):

        if i == (len(layer_specs)-1):
            #outermost layer
            del_val = h - y
            del_dict[i] = del_val
        else:
            #inner layers
            beta = del_val @ theta[i]
            #throw out bias term
            beta = beta[:,1:]
            del_val = beta * sigmoid_gradient(z_dict[i])
            del_dict[i] = del_val


        additive = (1/m) * (del_val.transpose() @ a_dict[i-1])
        grad_theta[i-1] = grad_theta[i-1] + additive
        reg_term = ((lam/m) * theta[i-1][:,1:])
        grad_theta[i-1][:,1:] = grad_theta[i-1][:,1:] + reg_term
        #grad_theta.set_theta_ind(grad_th[i-1], (i-1))
        #print('Type of grad_th:', type(grad_th), '\n', grad_th)
        #sets grad_theta object to the newly adjusted theta
    #====================================================

    return grad_theta

def check_gradient(theta, X, y, layer_specs, lam):
    offset = 0.00000001
    #set theta to theta flat
    #set an array of zeros, iterate through adding the offset, then run the numerical gradient
    #iterate through and add the offset, run the
    new_theta = Theta(layer_specs)
    new_theta.set_theta(theta.get_flat())
    delta_mat = Theta(layer_specs, 0)
    delta = delta_mat.get_flat()

    num_grad_mat = Theta(layer_specs, 0)
    num_grad = num_grad_mat.get_flat()

    for i in range(int(delta.size)):
        print('...checking gradient number: '+str(i+1)+'/'+str(delta.size), end='\r')
        # print('......running through each gradient.......', i, '/', delta.size)
        delta[i] = offset
        new_theta.set_theta(theta.get_flat() - delta)
        h, a_values, z_values = feed_forward(new_theta, X, layer_specs)
        loss1, cost_matrix = calculate_cost(new_theta, y, h, lam)

        new_theta.set_theta(theta.get_flat() + delta)
        h, a_values, z_values = feed_forward(new_theta, X, layer_specs)
        loss2, cost_matrix = calculate_cost(new_theta, y, h, lam)

        #reset new_theta after processing to theta.get_flat()
        new_theta.set_theta(theta.get_flat())

        #compute gradient here
        num_grad[i] = (loss2 - loss1) / (2 * offset)

        delta[i] = 0 #return to zeros all around
    num_grad_mat.set_theta(num_grad)

    return num_grad_mat

def grad_descent(theta, gradients, alpha):
    '''
    Takes one step in the gradient descent direction
    '''
    new_theta_flat = theta.get_flat() - alpha * gradients.get_flat()
    theta.set_theta(new_theta_flat)
    return theta

def select_slice(Xdata, Ydata, number_batches = 5, batch_number = 0):
    #start indexing at 0 -> n-1
    m = Xdata.shape[0]
    examples_per_batch = m // number_batches
    start_idx = batch_number * examples_per_batch
    end_idx = (batch_number + 1) * examples_per_batch

    #catch the end of the data set, if it's an uneven set
    if batch_number == number_batches - 1:
        end_idx = m

    x_slice = Xdata[start_idx:end_idx,:]
    y_slice = Ydata[start_idx:end_idx,:]

    return x_slice, y_slice

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

def test_vals():
    theta_contents = scipy.io.loadmat('ex4weights.mat')
    theta1 = theta_contents['Theta1']
    theta2 = theta_contents['Theta2']
    theta2_flat = theta2.flatten()
    theta1_flat = theta1.flatten()
    th = np.concatenate((theta1_flat, theta2_flat))
    test_theta = Theta(nn_specs)
    test_theta.set_theta(th)

    h_val, a_values, z_values = feed_forward(test_theta, x_mat, nn_specs)
    theta_train_grad = backprop(test_theta, y_mat, h_val, a_values, z_values, lam, nn_specs)
    cost, cost_matrix = calculate_cost(test_theta, y_mat, h_val, lam)
    print('cost at fixed debugging parameters w/ lambda =', lam, ':', cost)

def optimize_lambda():
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

def shuffle_data(x_mat, y_mat, sections = 'all'):
    #could later insert

    num_examples = x_mat.shape[0]
    value_indices = list(range(num_examples))
    random.shuffle(value_indices)
    if sections == 'all':
        end_index = num_examples
    else:
        end_index = int(num_examples / sections)

    shuffled_x_section = x_mat[value_indices[0:end_index], :]
    shuffled_y_section = y_mat[value_indices[0:end_index], :]

    return shuffled_x_section, shuffled_y_section

def load_data():
    mat_contents = scipy.io.loadmat('sampleData.mat')
    x_mat = mat_contents['X']
    y_vec = mat_contents['y']
    y_mat = logical_y_matrix(y_vec,10)
    return x_mat, y_mat

def split_data(x_all_data, y_all_data, train_fraction, validate_fraction, test_fraction):
    if round((train_fraction + validate_fraction + test_fraction),1) != 1:
        #this just requires the user to enter three fractions, though the test_fraction isn't used
        raise ValueError("your fractions don't add up to 1, please re-enter fractions.")


    train_end = int(train_fraction * y_all_data.shape[0])
    validate_start = train_end
    validate_end = validate_start + int(validate_fraction * y_all_data.shape[0])
    test_start = validate_end

    #split up the data into train, validate, and test
    x_train = x_all_data[:train_end, :]
    y_train = y_all_data[:train_end, :]
    x_validate = x_all_data[validate_start:validate_end, :]
    y_validate = y_all_data[validate_start:validate_end, :]
    x_test = x_all_data[test_start:, :]
    y_test = y_all_data[test_start:, :]

    return x_train, y_train, x_validate, y_validate, x_test, y_test

def train_nn(X, y, layer_specs, lam, max_iter = 20 , eps_limit = 0.1, numerical_check = None, descent_type = 'batch', number_batches = 5, batch_order = 'ordered', batch_mod_type = 'end', plot_input = False):
    '''
    train the network's activation weights

    '''
    #initialize a random theta
    theta_train = Theta(layer_specs)
    theta_train.generate_random_theta()

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
            cost, cost_matrix = calculate_cost(theta_train, y, h, lam)

        elif descent_type == 'mini-batch':
            if batch_order == 'ordered':
                i_batch = i % number_batches
            elif batch_order == 'random':
                i_batch = random.randint(0, (number_batches - 1))

            #choose the slice of data you want, based on batch number

            x_mb, y_mb = select_slice(X, y, number_batches, i_batch)
            h, a_values, z_values = feed_forward(theta_train, x_mb, layer_specs)
            theta_train_grad = backprop(theta_train, y_mb, h, a_values, z_values, lam, layer_specs)
            cost, cost_matrix = calculate_cost(theta_train, y_mb, h, lam)

        elif descent_type == 'mini-batch-mod':
            if batch_order == 'ordered':
                i_batch = i % number_batches
            elif batch_order == 'random':
                i_batch = random.randint(0, (number_batches - 1))
            #FINISH THIS
            #this starts with mini-batch, and then changes to batch for the last few iterations
            if batch_mod_type == 'end':
                if i < (max_iter - int(max_iter / 20)):
                    x_mb, y_mb = select_slice(X, y, number_batches, i_batch)
                    h, a_values, z_values = feed_forward(theta_train, x_mb, layer_specs)
                    theta_train_grad = backprop(theta_train, y_mb, h, a_values, z_values, lam, layer_specs)
                    cost, cost_matrix = calculate_cost(theta_train, y_mb, h, lam)
                elif i >= (max_iter - int(max_iter / 20)):
                    # perform full gradient descent for the last 1/20th of training iterations, to reach stability
                    h, a_values, z_values = feed_forward(theta_train, X, layer_specs)
                    theta_train_grad = backprop(theta_train, y, h, a_values, z_values, lam, layer_specs)
                    cost, cost_matrix = calculate_cost(theta_train, y, h, lam)

            elif batch_mod_type == 'mixed':
                #this isn't very effective
                if i % int(m / 10) != 0:
                    #this catches almost all of the iterations
                    x_mb, y_mb = select_slice(X, y, number_batches, i_batch)
                    h, a_values, z_values = feed_forward(theta_train, x_mb, layer_specs)
                    theta_train_grad = backprop(theta_train, y_mb, h, a_values, z_values, lam, layer_specs)
                    cost, cost_matrix = calculate_cost(theta_train, y_mb, h, lam)

                elif i % int(m / 10) == 0:
                    #this segment of normal gradient descent will happen 10 times during the training examples
                    for j in range(number_batches * 5):
                        h, a_values, z_values = feed_forward(theta_train, X, layer_specs)
                        theta_train_grad = backprop(theta_train, y, h, a_values, z_values, lam, layer_specs)
                        cost, cost_matrix = calculate_cost(theta_train, y, h, lam)
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

            num_grad = check_gradient(theta_train, x_check, y_check, layer_specs, lam)
            sum_difference = 0
            m_theta = theta_train_grad.get_flat().shape[0]

            for j in range(num_grad.get_flat().shape[0]):
                difference = (theta_train_grad.get_flat()[j] - num_grad.get_flat()[j])
                num_check.append([theta_train_grad.get_flat()[j], num_grad.get_flat()[j], difference])

            print('backprop gradient - numerical gradient - difference')
            for k in range(0, num_grad.get_flat().shape[0], int(num_grad.get_flat().shape[0]/20)):
                print(num_check[k])
            # average_difference = sum_difference / theta_train_grad.get_flat().shape[0]
            # print('average difference between numerical gradient and backprop gradient:\n', average_difference)
            num_check = np.array(num_check)
            #print('backprop gradient - numerical gradient - difference\n', num_check)
            input('press enter to continue training neural network weights.')

        #print('Theta before grad_descent:', theta_train.get_flat())
        #perform gradient descent, alter theta_train
        theta_train = grad_descent(theta_train, theta_train_grad, 2.0)
        #print('Theta after grad_descent:', theta_train.get_flat())
        #update while loop conditional values
        eps = (temp_cost - cost)
        print('cost after iteration '+ str(i) + ': ' + str(cost), end='\r')

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

        i += 1

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

x_mat, y_mat = load_data()
x_all_shuffled, y_all_shuffled = shuffle_data(x_mat, y_mat)

x_train, y_train, x_vali, y_vali, x_test, y_test = split_data(x_all_shuffled, y_all_shuffled, 0.9, 0.0, 0.1)

neural_net_specs = (400, 25, 10)

print('training neural network...')
trained_cost, trained_theta, trained_health, trained_eps = train_nn(x_train, y_train, neural_net_specs, lam = 0.05, max_iter = 1000, eps_limit = 0.1, numerical_check = None, descent_type = 'mini-batch-mod', number_batches = 7, batch_order = 'ordered', batch_mod_type = 'end', plot_input = True)

evaluate_nn(trained_theta, x_test, y_test, neural_net_specs)
