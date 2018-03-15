import scipy, scipy.io
import numpy as np
#import math
#import sklearn as sk
#NOTES: * is .* , and @ is matrix multiplication

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
    logicalArray = np.zeros((vectorY.shape[0],numberLabels))
    for i in range(logicalArray.shape[0]):
        logicalArray[i,(vectorY[i]-1)] = 1
        #print(logicalArray[i])
    return logicalArray



def initializeRandomTheta(netSpecs, eps = 0.12):
    '''
    This initializes a long array of random values for all the theta matrices
    Input: nodeSpecs(2,4,3) as before
    '''
    numValues = 0
    for i in range(len(netSpecs)-1):
        numValues += ((nodeSpecs[i]+1)*nodeSpecs[i+1])
    randomTheta = (np.random.rand(numValues,1))*2*eps - eps
    return randomTheta


def cost_and_gradient(unrolledTheta, X, y, lam, num_labels, layerSpecs):
    '''
    Inputs:
    X: [m x n] matrix, m-number of training examples, n-number of inputs (features)
    y: [m x 1] vector, m-training example outputs (tags)
    unrolledTheta = unrolled combination of matrices for all of the layers, unrolled by rows
    numLayers: integer, number of layers, including first and last
    layerSpecs: tuple of number of number of nodes in each layer, including first and last
                i.e. [3, 4, 2] - 3 inputs, 4 nodes in hidden layer, 2 output nodes
    ----------------------------------
    Outputs:
    cost = floating point number of cost based on current values of theta
    gradient = array of gradients for theta

    -------------
    a = cost_and_gradient(........)
    a[0] - returns the cost function
    a[1] - returns the gradient
    -------------

    ALL INDICES START FROM 0

    '''

    #CHECK IF THE NUMBER OF LABELS MATCHES THE NUMBER OF OUTPUT NODES
    if num_labels != layerSpecs[-1]:
        print("We've got a problem, num_labels and the number of output nodes don't match. \nPlease check your inputs and retry.")
        return
    #finds matrix sizes, based on number of nodes in neuralnet
    #----list of tuples, (numRows, numColumns)
    #THIS WORKS
    matrixSizes = []
    for i in range(len(layerSpecs)-1):
        matrixSizes.append((layerSpecs[i+1], (layerSpecs[i]+1)))

    m = X.shape[0]
    #transforms unrolledTheta into a list of theta matrices
    #also creates a list of grad Theta matrices
    #THIS WORKS
    thetaList = []
    gradThetaList = []
    temp = 0
    for specs in matrixSizes:
        numRow = specs[0]
        numCol = specs[1]
        numElements = numRow * numCol
        #print('number of rows:', numRow, ' number of cols:', numCol , '  number of elements', numElements)
        rowMatrix = unrolledTheta[temp:(temp+numElements)]
        #print('size of row matrix', len(rowMatrix), '  row matrix:', rowMatrix)
        temp += numElements
        thetaList.append(np.reshape(rowMatrix, (numRow, numCol)))
        gradThetaList.append(np.zeros((numRow, numCol)))

    #calculate Hypothesis, h:
    aList = [] #list of the nodes
    zList = [] #list of the intermediate values
    #find value of hypothesis, h (loop based on number of layers), calculate, a, z
    #Feed forward programming
    for layer in range(len(layerSpecs)):
        if layer == 0: #special options for first iteration
            print('first layer')
            # xShape = np.shape(X)
            # print(xShape)
            ones = (np.ones((np.shape(X)[0],1)))
            aVal = np.append(ones,X, axis = 1)
            aList.append(aVal)
        elif layer == (len(layerSpecs)-1):
            print('last layer')
            zVal = aList[(layer-1)] @ thetaList[(layer-1)].transpose()
            aVal = sigmoid(zVal)
            aList.append(aVal)
            h = aVal
        else: #for all the middle layers
            print('middle layer')
            zVal = aList[(layer-1)] @ thetaList[(layer-1)].transpose()
            zList.append(zVal)
            #calculate the sigmoid of the zValue
            aVal = sigmoid(zVal)
            #add a column of ones since it's a middle layer, accounting for bias unit
            ones = (np.ones((np.shape(aVal)[0],1)))
            aVal = np.append(ones,aVal, axis = 1)
            aList.append(aVal)

    #calculating the cost with the current parameters
    #==================================================
    jMatrix = ((-y * np.log(h))-((1-y)*np.log(1-h)))
    jTheta = (1/m)*np.sum(jMatrix)

    #adding in regularized terms, omitting theta bias terms
    theta_sums = 0
    for theta_i in range(len(thetaList)):
        theta_squared = np.square(thetaList[theta_i][:,1:])
        theta_sums += np.sum(theta_squared)
    jTheta += (lam/(2*m))*theta_sums
    #==================================================

    #calculating the gradients based on the current parameters
    #==================================================
    delList = []
    for i in range((len(layerSpecs)-1),0,-1):
        if i == (len(layerSpecs)-1):
            #outermost step
            del_val = h - y
            delList.append(del_val)

        else:
            beta = del_val @ thetaList[i]
            #throw out bias terms
            beta = beta[:,1:] #reduce dimension of beta by one column
            del_val = beta * sigmoidGradient(zList[i-2])
            delList.insert(0, del_val)

        #THIS ADDITION HAPPENS AFTER ALL STEPS
        gradThetaList[i-1] += del_val.transpose() @ aList[i-1]
        gradThetaList[i-1] = (1/m) * gradThetaList[i-1]
        gradThetaList[i-1][:,1:] + ((lam/m) * thetaList[i-1][:,1:])


    #==================================================

    # #TYPE AND DIMENSION CHECKERS
    # print('size X:', X.shape)
    # #print('type of X:', type(X))
    # for i in range(len(aList)):
    #     print('size of matrix a',i,':', aList[i].shape)
    #     #print('type of matrix a',i,':', type(aList[i]))
    # for i in range(len(zList)):
    #     print('size of matrix z',i,':', zList[i].shape)
    #     #print('type of matrix z',i,':', type(zList[i]))
    # for i in range(len(thetaList)):
    #     print('size of matrix theta',i,':', thetaList[i].shape)
    #     #print('type of matrix z',i,':', type(zList[i]))
    # for i in range(len(gradThetaList)):
    #     print('size of matrix gradTheta',i,':', gradThetaList[i].shape)
    #     #print('type of matrix z',i,':', type(zList[i]))
    # for i in range(len(delList)):
    #     print('size of matrix del',i,':', delList[i].shape)
    #     #print('type of matrix z',i,':', type(zList[i]))
    # print('size of h:', h.shape)
    # print('size of jMatrix:', jMatrix.shape)
    # #print('jMatrix:\n', jMatrix)
    # print('J(theta):', jTheta)

    return jTheta, gradThetaList
# mat_contents= scipy.io.loadmat('sampleData.mat')
# xMatrix = mat_contents['X']
# yVector = mat_contents['y']

#these have to have the same number of rows!!!!
# y_ = np.array([[1],[3],[2],[1]])
# y = logicalYMatrix(y_,3)
# x = np.array([[0.3, 1],[2, 7],[99, 1],[2, 0.6]])

mat_contents = scipy.io.loadmat('sampleData.mat')
x_mat = mat_contents['X']
y_vec = mat_contents['y']
y_mat = logicalYMatrix(y_vec,10)

theta_contents = scipy.io.loadmat('ex4weights.mat')
theta1 = theta_contents['Theta1']
theta2 = theta_contents['Theta2']
theta2_flat = theta2.flatten()
theta1_flat = theta1.flatten()


theta = np.concatenate((theta1_flat, theta2_flat))



# theta2_reshape = np.reshape(theta2_flat, (10,26))
# print('===========================')
# print(theta2)
# print('===========================')
# print(theta2_reshape)
# print('===========================')
# print(theta2.shape)
# print(theta2_reshape.shape)



nodeSpecs = (400,25,10)
numLabels = nodeSpecs[-1]
#theta = initializeRandomTheta(nodeSpecs)
lam = 1
C_G = cost_and_gradient(theta, x_mat, y_mat, lam, numLabels, nodeSpecs)
print(C_G[0])
print(C_G[1])

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;
node
