import scipy, scipy.io
import numpy as np
import pandas as pd

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

    def generate_random_theta(self, eps = 0.12):
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

def feed_forward(theta, X, y, layer_specs):
    m = y.shape[0]

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

def check_gradient(theta, X, y, layer_specs, lam = 3): #THIS DOESN'T WORK YET
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
        delta[i] = offset
        new_theta.set_theta(theta.get_flat() - delta)
        h, a_values, z_values = feed_forward(new_theta, X, y, layer_specs)
        loss1, cost_matrix = calculate_cost(new_theta, y, h, lam)


        new_theta.set_theta(theta.get_flat() + delta)
        h, a_values, z_values = feed_forward(new_theta, X, y, layer_specs)
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

def train_nn(X, y, layer_specs, lam, max_iter = 20 , eps_limit = 0.1, numerical_check = False):
    '''
    train the network's activation weights

    '''
    #initialize a random theta
    theta_train = Theta(layer_specs)
    theta_train.generate_random_theta()

    #feed forward, backpropagate and calculate cost
    #keep iterating until you pass max_iter or eps drops below eps_limit
    i = 0
    eps = 1000 # default
    temp_cost = 100000000000000
    while (i <= max_iter):
        h, a_values, z_values = feed_forward(theta_train, X, y, layer_specs)
        theta_train_grad = backprop(theta_train, y, h, a_values, z_values, lam, layer_specs)
        cost, cost_matrix = calculate_cost(theta_train, y, h, lam)
        #print('Theta before grad_descent:', theta_train.get_flat())
        #perform gradient descent, alter theta_train
        theta_train = grad_descent(theta_train, theta_train_grad, 0.3)
        #print('Theta after grad_descent:', theta_train.get_flat())
        #update while loop conditional values
        eps = (temp_cost - cost)
        temp_cost = cost #set temp_cost for next iteration
        i += 1

        if numerical_check == True and i == 1:
            #only checks the gradient on the first iteration
            num_grad = check_gradient(theta_train, X, y, layer_specs, lam)
            sum_difference = 0
            for i in range(theta_train_grad.get_flat().shape[0]):
                sum_difference += (theta_train_grad.get_flat()[i] - num_grad.get_flat()[i])
            average_difference = sum_difference / theta_train_grad.get_flat().shape[0]
            print('average difference between numerical gradient and backprop gradient:\n', average_difference)

        print('cost after iteration', i-1, ':', cost)
        print('eps:', eps)

var_Data = pd.read_csv("lossData.csv", float_precision='round_trip')
xyData = pd.read_csv("XYData.csv", float_precision='round_trip')

#test parameters to check against what my operations should actually result in.
var_Data = np.array(var_Data)
testTheta = var_Data[:,3]
testGrad = var_Data[:,4]
test_specs = (3,5,3)
test_theta = Theta(test_specs)
test_theta.set_theta(testTheta.transpose())
test_grad = Theta(test_specs) #this might be wrong
test_grad.set_theta(testGrad.transpose())
xyData = np.array(xyData)
test_X = np.array(xyData[0:,:3])
test_y = np.array((xyData[0:,3]))
print()
test_y = logical_y_matrix(test_y, 3)
test_lam = 3

# h, a_values, z_values = feed_forward(test_theta, test_X, test_y, test_specs)
# theta_train_grad = backprop(test_theta, test_y, h, a_values, z_values, test_lam, test_specs)
# cost, cost_matrix = calculate_cost(test_theta, test_y, h, test_lam)
#
# print(cost)

train_nn(test_X, test_y, test_specs, test_lam, max_iter = 1000, numerical_check = False)
