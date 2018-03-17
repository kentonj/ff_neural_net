import scipy, scipy.io
import numpy as np
#import math
#import sklearn as sk
#NOTES: * is .* , and @ is matrix multiplication

class Theta(object):
    #requires inputs of neural net specs (2,4,3)--number of nodes in each layer
    #optional input of the theta
    def __init__(self, neural_net_specs):
        #theta set to 0 as default
        self.specs = neural_net_specs
        #initialize random theta
        self.theta_dict = {}

    def compile_theta(self, row_v):
        temp_index = 0
        #copied so that it does not point to the same location, and cannot be changed
        #accidentally from outside of the class
        row_vec = np.copy(row_v)
        for i in range(len(self.specs)-1):
            theta_dims = (self.specs[i+1], self.specs[i]+1)
            num_values = theta_dims[0] * theta_dims[1]
            self.theta_dict[i] = np.reshape(row_vec[temp_index:(temp_index+num_values)],theta_dims)
            temp_index += num_values

    def generate_random_theta(self, eps = 0.12):
        #generates a dictionary of random theta values, with a dictionary key
        #correlating to it's position in the neural network
        for i in range(len(self.specs)-1):
            theta_dims = (self.specs[i+1], self.specs[i]+1)
            random_theta = (np.random.rand(theta_dims[0],theta_dims[1]))*2*eps - eps
            self.theta_dict[i] = random_theta

    def flatten_theta(self):
        #flattens theta matrix into row array
        for i_theta in range(len(self.theta_dict)):
            if i_theta == 0:
                self.flattened_theta = self.theta_dict[i_theta].flatten()
            else:
                self.flattened_theta = np.concatenate((self.flattened_theta, self.theta_dict[i_theta].flatten()))


    def get_theta_flat(self):
        #makes sure that it reflattens theta, before returning the flattened theta
        #this makes sure that it returns the most recent theta
        Theta.flatten_theta(self)
        return self.flattened_theta

    def required_theta_dims(self, dict_key):
        dict_key = int(dict_key) #forces it into an integer
        if dict_key > (len(self.specs) - 1): #catches theta outside of the acceptable range
            raise TypeError('You entered a theta value that is outside of the limits of this neural net.')
        theta_dims = (self.specs[dict_key + 1], (self.specs[dict_key]+1))
        return theta_dims

    def set_individual_theta(self, dict_key, new_theta):
        if type(new_theta) == list:
            #converts an input list to a NumPy array
            new_theta = np.array(new_theta)
        required_dims = Theta.required_theta_dims(self, dict_key) #generates the acceptable size

        #checks if there is an existing dictionary with the same key
        if dict_key in self.theta_dict.keys():
            if new_theta.shape == required_dims: #new theta is same shape as expected shape
                self.theta_dict[dict_key] = new_theta
            elif (required_dims[0] * required_dims[1]) == new_theta.size: #row vector
                self.theta_dict[dict_key] = np.reshape(new_theta, required_dims)
            else:
                raise TypeError('Incorrect size or shape for theta, based on the neural net specs.')

        else:
            #if the dictionary key doesn't exist, then make sure that it is either the right shape or size
            if new_theta.shape == required_dims: #new theta is same shape as expected shape
                self.theta_dict[dict_key] = new_theta
            elif (required_dims[0] * required_dims[1]) == new_theta.size: #row vector
                self.theta_dict[dict_key] = np.reshape(new_theta, required_dims)
            else:
                raise TypeError('Incorrect size or shape for theta, based on the neural net specs.')

    def set_all_theta(self, row_vector):
        #this resets all theta, requires a row vector or a list
        if type(row_vector) == list:
            #convert into a numpy array, if the input is a list
            row_vector = np.array(row_vector)
        count = 0
        for i in range(len(self.specs)-1):
            dims = Theta.required_theta_dims(self, i)
            count += (dims[0] * dims [1])
        if row_vector.size != count:
            raise TypeError('Existing thetas and new theta have different number of values.')
        temp_index = 0
        for i in range(len(self.specs)-1):
            theta_shape = Theta.required_theta_dims(self, i)
            num_values = theta_shape[0] * theta_shape[1]
            self.theta_dict[i] = np.reshape(row_vector[temp_index:(temp_index+num_values)],theta_shape)
            temp_index += num_values

    def zeros(self):
        for i in range(len(self.specs)-1):
            self.theta_dict[i] = np.zeros((self.specs[i+1], (self.specs[i]+1)))

    def get_theta_matrices(self):
        return self.theta_dict

    def __str__(self):
        return str(self.theta_dict)

def sigmoid(z):
    sig = (1.0 /(1.0+np.exp(-1.0 * z)))
    return sig

def sigmoidGradient(z):
    sigGrad = sigmoid(z)*(1-sigmoid(z))
    return sigGrad

def logicalYMatrix(vectorY, numberLabels):
    '''
    creates a logical array for classification:
    input: [[2],[3],[1]]
    output: [[0,1,0],[0,0,1],[1,0,0]]
    '''
    if type(vectorY) == list:
        vectorY = np.array(vectorY)

    logicalArray = np.zeros((vectorY.shape[0],numberLabels))
    for i in range(logicalArray.shape[0]):
        logicalArray[i,(vectorY[i]-1)] = 1
        #print(logicalArray[i])
    return logicalArray



def cost_and_grad(th, X, y, layer_specs, lam = 1):
    '''

    Description: this function computes the cost and the gradient of a neural net

    INPUTS:
    theta: Theta object defined above, weights for each layer
    X: list or NumPy array of inputs, organized by columns being each x, rows being each example
    y: NumPy logical array, rows of examples
    layer_specs: (2, 4, 3) - (inputs, hidden layer nodes, outputs)
    lam: Optional input, defines theta, normalization factor

    OUTPUTS:
    jTheta: Cost based on current weights
    gradient: Gradients for each weight

    '''

    #convert X to a numpy array if needed
    if type(X) == list:
        X = np.array(X)
    m = X.shape[0]
    #converts theta input and retrieves the theta_dict
    theta = th.get_theta_matrices()


    #FEED FORWARD
    #====================================================
    a_dict = {} #dict of the nodes
    z_dict = {} #dict of the intermediate values
    #find value of hypothesis, h (loop based on number of layers), calculate, a, z
    #Feed forward programming
    for layer in range(len(layer_specs)):
        if layer == 0: #special options for first iteration
            #print('first layer')
            ones = (np.ones((np.shape(X)[0],1)))
            aVal = np.append(ones, X, axis = 1)
            a_dict[layer] = aVal
        elif layer == (len(layer_specs)-1):
            #print('last layer')
            zVal = a_dict[(layer-1)] @ theta[(layer-1)].transpose()
            aVal = sigmoid(zVal)
            a_dict[layer] = aVal
            h = aVal
        else: #for all the middle layers
            #print('middle layer')
            zVal = a_dict[(layer-1)] @ theta[(layer-1)].transpose()
            z_dict[layer] = zVal
            #calculate the sigmoid of the zValue
            aVal = sigmoid(zVal)
            #add a column of ones since it's a middle layer, accounting for bias unit
            ones = (np.ones((np.shape(aVal)[0],1)))
            aVal = np.append(ones, aVal, axis = 1)
            a_dict[layer] = aVal
    #====================================================

    #CALCULATE COST BASED ON CURRENT THETA
    #====================================================
    jMatrix = ((-y * np.log(h))-((1-y)*np.log(1-h)))
    jTheta = (1/m)*np.sum(jMatrix)

    #adding in regularized terms, omitting theta bias terms
    theta_sums = 0
    for theta_i in range(len(theta)):
        theta_sums += np.sum(np.square(theta[theta_i][:,1:]))
    jTheta += (lam/(2*m))*theta_sums
    #====================================================


    #BACK PROPOGATION
    #initialize a gradient theta object
    #====================================================
    grad_theta = Theta(layer_specs)
    grad_theta.zeros()
    grad_th = grad_theta.get_theta_matrices()

    del_dict = {}
    for i in range((len(layer_specs)-1), 0, -1):

        if i == (len(layer_specs)-1):
            #outmost layer
            del_val = h - y
            del_dict[i] = del_val
        else:
            #inner layers
            beta = del_val @ theta[i]
            #throw out bias term
            beta = beta[:,1:]
            del_val = beta * sigmoidGradient(z_dict[i])
            del_dict[i] = del_val
        # print('gradient shape:', grad_th[i-1].shape)
        # print('del_val shape:', del_val.shape)
        # print('a_dict shape:', a_dict[i-1].shape)
        grad_th[i-1] += del_val.transpose() @ a_dict[i-1]
        grad_th[i-1] = (1/m) * grad_th[i-1]
        grad_th[i-1][:,1:] + ((lam/m) * theta[i-1][:,1:])
    #====================================================


    return jTheta, grad_theta




mat_contents = scipy.io.loadmat('sampleData.mat')
x_mat = mat_contents['X']
y_vec = mat_contents['y']
y_mat = logicalYMatrix(y_vec,10)

theta_contents = scipy.io.loadmat('ex4weights.mat')
theta1 = theta_contents['Theta1']
theta2 = theta_contents['Theta2']
theta2_flat = theta2.flatten()
theta1_flat = theta1.flatten()


th = np.concatenate((theta1_flat, theta2_flat))

node_spec = (400, 25, 10)

theta_nn = Theta(node_spec)
theta_nn.set_all_theta(th)

C_G = cost_and_grad(theta_nn, x_mat, y_mat, node_spec)
print('Cost:', C_G[0])
print('Gradients:', C_G[1])
