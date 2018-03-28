import scipy, scipy.io
import numpy as np
import pandas as pd



#import math
#import sklearn as sk
#NOTES: * is .* , and @ is matrix multiplication




def sigmoid(z):
    sig = (1.0 /(1.0+np.exp(-1.0 * z)))
    return sig

def sigmoidGradient(z):
    sigGrad = sigmoid(z)*(1-sigmoid(z))
    return sigGrad





def cost_and_grad(theta, X, y, layer_specs, lam = 1):
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

    if type(y) == list:
        y = np.array(y)
    if y.shape[1] == 1:
        #this means it isn't a logical array yet
        y = logicalYMatrix(y)

    #FEED FORWARD
    #====================================================
    a_dict = {} #dict of the nodes
    z_dict = {} #dict of the intermediate values
    #find value of hypothesis, h (loop based on number of layers), calculate, a, z

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

    #CALCULATE COST BASED ON CURRENT THETA
    #====================================================
    jMatrix = ((-y * np.log(h))-((1-y)*np.log(1-h)))
    jTheta = (1/m)*np.sum(jMatrix)

    #adding in regularized terms, omitting theta bias terms
    #====================================================
    theta_sums = 0
    for theta_i in range(len(theta.matrices)):
        theta_sums += np.sum(np.square(theta[theta_i][:,1:]))
    jTheta += (lam/(2*m))*theta_sums
    #====================================================


    #BACK PROPOGATION
    #initialize a gradient theta object
    #====================================================
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
            del_val = beta * sigmoidGradient(z_dict[i])
            del_dict[i] = del_val


        additive = (1/m) * (del_val.transpose() @ a_dict[i-1])
        grad_theta[i-1] = grad_theta[i-1] + additive
        reg_term = ((lam/m) * theta[i-1][:,1:])
        grad_theta[i-1][:,1:] = grad_theta[i-1][:,1:] + reg_term
        #grad_theta.set_theta_ind(grad_th[i-1], (i-1))
        #print('Type of grad_th:', type(grad_th), '\n', grad_th)
        #sets grad_theta object to the newly adjusted theta


    #====================================================
    grad_theta.flatten_theta()

    return jTheta, grad_theta



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
        logicalArray[i,(int(vectorY[i])-1)] = 1.0

    return logicalArray

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
        Theta.flatten_theta(self)

    def flatten_theta(self):
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
        Theta.flatten_theta(self)
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
        Theta.flatten_theta(self)
        #return self.matrices

    def get_theta_matrices(self):
        return self.matrices

    def __getitem__(self, key):

        return self.matrices[key]

    def __setitem__(self, key, item):

        self.matrices[key] = item

    def __str__(self):
        return str(self.matrices)

def check_gradient(theta, X, y, layer_specs, lam = 3): #THIS DOESN'T WORK YET
    offset = 0.000000001
    #set theta to theta flat
    #set an array of zeros, iterate through adding the offset, then run the numerical gradient
    #iterate through and add the offset, run the

    new_theta = Theta(layer_specs)
    new_theta.set_theta(theta.flat)
    delta_mat = Theta(layer_specs, 0)
    delta = delta_mat.flat

    num_grad_mat = Theta(layer_specs, 0)
    num_grad = num_grad_mat.flat

    for i in range(int(delta.size)):
        delta[i] = offset
        new_theta.set_theta(theta.flat - delta)
        loss1 = cost_and_grad(new_theta, X, y, layer_specs, lam)[0]

        new_theta.set_theta(theta.flat + delta)
        loss2 = cost_and_grad(new_theta, X, y, layer_specs, lam)[0]

        #reset new_theta after processing to theta.flat
        new_theta.set_theta(theta.flat)

        #compute gradient here
        num_grad[i] = (loss2 - loss1) / (2 * offset)

        delta[i] = 0 #return to zeros all around
    num_grad_mat.set_theta(num_grad)

    return theta, num_grad_mat




# mat_contents = scipy.io.loadmat('sampleData.mat')
# x_mat = mat_contents['X']
# y_vec = mat_contents['y']
# y_mat = logicalYMatrix(y_vec,10)
#
# theta_contents = scipy.io.loadmat('ex4weights.mat')
# theta1 = theta_contents['Theta1']
# theta2 = theta_contents['Theta2']
# theta2_flat = theta2.flatten()
# theta1_flat = theta1.flatten()
#
#
# th = np.concatenate((theta1_flat, theta2_flat))
#
# node_spec = (400, 25, 10)
#
# theta_nn = Theta(node_spec)
# theta_nn.set_theta(th)
#
# C_G = cost_and_grad(theta_nn, x_mat, y_mat, node_spec)
# print(C_G[0])

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
test_y = logicalYMatrix(test_y, 3)
lambda_test = 3
J, grad_nn = cost_and_grad(test_theta, test_X, test_y, test_specs, lambda_test)
#theta, X, y, layer_specs, lam = 1



# I THINK I AM LOSING SOME FIDELITY WHEN GOING FROM PANDAS into NUMPY
output_theta, num_gradient = check_gradient(test_theta, test_X, test_y, test_specs)
print('numerical gradient:\n', num_gradient.flat)
print('backprop gradient:\n', grad_nn.flat)