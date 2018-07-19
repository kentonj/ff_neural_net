import scipy, scipy.io
import numpy as np
import random

class WeightMatrices(object):
    #WeightMatrices BASE REPRESENTATION IS A LONG ROW VECTOR
    #using get_matrix(key) returns the specific matrix in question as a "view" of the entire row
    def __init__(self, net_specs, initialization = 'random', row_entry = [0]):
        self.net_specs = net_specs
        self.dimensions = WeightMatrices.get_mat_dims(self)
        self.count = WeightMatrices.num_weights(self)
        self.start_end_dict = WeightMatrices.start_end(self)

        #might be a more elegant way to do this
        if len(row_entry) == 1:
            self.row = WeightMatrices.initialize_WeightMatrices(self, initialization)
        else:
            self.row = row_entry

    def initialize_WeightMatrices(self, initialization, eps = 0.12):
        if initialization == 'random':
            self.row = np.random.rand(self.count) * 2 * eps - eps
        elif initialization == 'zeros':
            self.row = np.zeros(self.count)
        elif initialization == 'ones':
            self.row = np.ones(self.count)
        return self.row

    def get_mat_dims(self):
        self.dimensions = {}
        for i in range(len(self.net_specs) - 1):
            self.dimensions[i] = ((self.net_specs[i+1]),(self.net_specs[i]+1))
        return self.dimensions

    def num_weights(self):
        self.count = 0
        for key in self.dimensions:
            self.count += self.dimensions[key][0] * self.dimensions[key][1]
        return self.count

    def start_end(self):
        self.start_end_dict = {}
        start = 0
        for key in self.dimensions:
            number_items = self.dimensions[key][0] * self.dimensions[key][1]
            end = start + number_items
            self.start_end_dict[key] = [start, end]
            start = end
        return self.start_end_dict

    def get_matrix(self, key):
        row_slice = self.row[self.start_end_dict[key][0]:self.start_end_dict[key][1]]
        matrix = np.reshape(row_slice, self.dimensions[key])
        return matrix

    def set_matrix(self, key, new_mat):
        new_row = new_mat.ravel()
        self.row[self.start_end_dict[key][0]:self.start_end_dict[key][1]] = new_row

    def get_row(self):
        return self.row

    def __str__(self):
        string = 'WeightMatrices Matrices:\n'
        for matrix in self.dimensions:
            string += ('WeightMatrices ' + str(matrix) + ' shape: ' + str(WeightMatrices.get_matrix(self, matrix).shape) + '\n' + str(WeightMatrices.get_matrix(self, matrix)) + '\n\n')
        return string

    def __getitem__(self, key):
        return self.row[key]

    def __setitem__(self, key, item):
        self.row[key] = item

    def __add__(self, other):
        total = self.row + other.row
        return WeightMatrices(self.net_specs, row_entry = total)

    def __sub__(self, other):
        total = self.row - other.row
        return WeightMatrices(self.net_specs, row_entry = total)
