import scipy, scipy.io
import numpy as np
#import math
#import sklearn as sk
#NOTES: * is .* , and @ is matrix multiplication

class Theta(object):
    #requires inputs of neural net specs (2,4,3)--number of nodes in each layer
    #optional input of the theta
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
        Theta.flatten_theta(self)

    def flatten_theta(self):
        for i_theta in range(len(self.matrices)):
            if i_theta == 0:
                self.flat = self.matrices[i_theta].flatten()
            else:
                self.flat = np.concatenate((self.flat, self.matrices[i_theta].flatten()))

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
        Theta.flatten_theta(self)

    def required_theta_dims(self, dict_key):
        dict_key = int(dict_key) #forces it into an integer
        if dict_key > (len(self.specs) - 1): #catches theta outside of the acceptable range
            raise TypeError('You entered a theta value that is outside of the limits of this neural net.')
        theta_dims = (self.specs[dict_key + 1], (self.specs[dict_key]+1))
        return theta_dims

    def set_zeros(self):
        for i in range(len(self.specs)-1):
            self.matrices[i] = np.zeros((self.specs[i+1], (self.specs[i]+1)))

    def get_theta_matrices(self):
        return self.matrices

    def __str__(self):
        return str(self.matrices)


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
    #converts theta input and retrieves the matrices
    theta = th.matrices


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
    grad_theta.set_zeros()
    grad_th = grad_theta.matrices

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



def check_gradient(theta, X, y, layer_specs, lam = 1):
    offset = 0.000001
    #set theta to theta flat
    #set an array of zeros, iterate through adding the offset, then run the numerical gradient
    #iterate through and add the offset, run the


    flat_theta = theta.get_theta_flat()
    new_theta = Theta(layer_specs)
    new_theta.set_all_theta(flat_theta)
    zeros = np.zeros((1,theta.get_theta_flat().size)).flatten()
    gradient_checked = np.zeros((1,theta.get_theta_flat().size)).flatten()
    for i in range(int((zeros.size)/60)):
        zeros[i] = offset
        new_theta.set_all_theta(flat_theta-zeros)
        (loss1, grad) = cost_and_grad(new_theta, X, y, layer_specs, lam)
        new_theta.set_all_theta(flat_theta+zeros)
        (loss2, grad) = cost_and_grad(new_theta, X, y, layer_specs, lam)

        gradient_checked[i] = (loss2 - loss1) / (2 * offset)

        zeros[i] -= offset #return to zeros all around


    return gradient_checked


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
theta_nn.set_theta(th)

C_G = cost_and_grad(theta_nn, x_mat, y_mat, node_spec)
flat_grad = C_G[1].flat
# print('Cost:', C_G[0])
# print('Gradients:', flat_grad[1:30])
#
# checked_gradient = check_gradient(theta_nn, x_mat, y_mat, node_spec)
# print(checked_gradient[1:30])

import csv
oData = []
with open('OctaveData.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        oData.append(row)
#turn oData into a numpy array
#test parameters
oData = np.array(oData)
testTheta = oData[1:,0]
testGrad = oData[1:,1]
test_specs = (3,5,3)
correct_theta = Theta(test_specs)
correct_theta.set_theta(testTheta.transpose())
print(correct_theta.flat)
correct_grad = Theta(test_specs)
correct_grad.set_theta(testGrad.transpose())
print(correct_grad.flat)