import scipy, scipy.io
import numpy as np
import random
from weightclass import Theta

def shuffle_data(x_mat, y_mat):
    #could later insert

    num_examples = x_mat.shape[0]
    value_indices = list(range(num_examples))
    random.shuffle(value_indices)
    end_index = num_examples

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

def select_slice(Xdata, Ydata, batch_number = 0, number_batches = 5):
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

def load_sample_theta(net_specs):
    sample_theta = Theta(net_specs, 'zeros')
    contents = scipy.io.loadmat('ex4weights.mat')
    theta0 = contents['Theta1']
    theta1 = contents['Theta2']
    sample_theta.set_matrix(0, theta0)
    sample_theta.set_matrix(1, theta1)
    return sample_theta
