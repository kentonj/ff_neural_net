import scipy, scipy.io
import numpy as np
import random
import os
from PIL import Image
from weightclass import WeightMatrices

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

def split_data(x_all_data, y_all_data, split_ratios):
    train_fraction, validate_fraction, test_fraction = split_ratios
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

def check_test_vali_error(va_error_list, lookback = 10):
    if len(va_error_list) < 3 * lookback:
        return False
    else:
        last_batch = len(va_error_list) - lookback
        second_last_batch = last_batch - lookback
        last_vali = va_error_list[last_batch:]
        second_last_vali = va_error_list[second_last_batch:last_batch]
        if np.mean(last_vali) > np.mean(second_last_vali):
            print('stopping training because validation error is now increasing.')
            return True
        else:
            return False

def predict_values(neural_net, testing_x, cutoff = 0.5, print_check = False):
    #neural_net is the network object's name (FeedForwardNN)
    m = testing_x.shape[0]
    h = neural_net.feedforward(neural_net.theta, testing_x)

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

def evaluate_nn(neural_net, x_test, y_test, cutoff = 0.5):
    #reformat y_test from 0 0 0 0 1 0 0 0 0  to 4
    y_correct = []
    for i in range(y_test.shape[0]):
        max_index = np.argmax(y_test[i,:])
        y_correct.append([max_index])
    y_correct = np.array(y_correct)

    output_vals = predict_values(neural_net, x_test, cutoff)
    number_wrong = 0
    for i in range(y_test.shape[0]):
        if y_correct[i,0] != output_vals[i,0]:
            number_wrong += 1
    m = y_correct.shape[0]
    print('total number incorrectly labeled:', number_wrong)
    print('total test examples:', m)
    print('percent correct:', round((((m - number_wrong) / m) * 100), 4))
    percent_correct = round((((m - number_wrong) / m) * 100), 4)
    return percent_correct

def convertPNGtoBW(png_file_location):
    img = Image.open(png_file_location).convert('LA')
    img.save('greyscale.png')

def resize_png(img, basewidth):
    basewidth = 300
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('sompic.jpg')
    return png.resize(dimensions, )

def convertallimagestoBW(top_level_folder, image_extension):
    directories = [x[0] for x in os.walk(top_level_folder)]
    for folder in directories:
        if 'train' in str(folder):
            #accumulate the training examples
            if 'NORMAL' in str(folder):
                for f in listdir(directory):
                    if f.endswith('.' + image_extension):
                        pass
                        #read file into black and white image
                        #compress to certain number of pixels
                        #write to csv with y value set to 0
                y = 0
            elif 'PNEUMONIA' in str(folder):
                for f in listdir(directory):
                    if f.endswith('.' + image_extension):
                        pass
                        #read file into black and white image
                        #compress to certain number of pixels
                        #write to csv with y value set to 0
                y = 1
        elif 'test' in str(folder):
            #accumulate test examples
            if 'NORMAL' in str(folder):
                for f in listdir(directory):
                    if f.endswith('.' + image_extension):
                        pass
                        #read file into black and white image
                        #compress to certain number of pixels
                        #write to csv with y value set to 0
                y = 0
            elif 'PNEUMONIA' in str(folder):
                for f in listdir(directory):
                    if f.endswith('.' + image_extension):
                        pass
                        #read file into black and white image
                        #compress to certain number of pixels
                        #write to csv with y value set to 0
                y = 1
        elif 'val' in str(folder):
            #accumulate validation examples
            if 'NORMAL' in str(folder):
                for f in listdir(directory):
                    if f.endswith('.' + image_extension):
                        pass
                        #read file into black and white image
                        #compress to certain number of pixels
                        #write to csv with y value set to 0
                y = 0
            elif 'PNEUMONIA' in str(folder):
                for f in listdir(directory):
                    if f.endswith('.' + image_extension):
                        pass
                        #read file into black and white image
                        #compress to certain number of pixels
                        #write to csv with y value set to 0
                y = 1

#convertallimagestoBW('/Users/kentoncozart/Documents/Datasets/chest_xray', 'png')
