import numpy as np
import math
import random
import sys


class Perceptron:
    layers = None

    learning_rate = 0.1
    max_error = 0.0001


    def __init__(self, nodes_count_per_layer):
        self.layers = []
        for k in range(len(nodes_count_per_layer) - 1):
            layer = []
            for j in range(nodes_count_per_layer[k]):
                layer.append(Node(nodes_count_per_layer[k + 1]))
            self.layers.append(layer)

    @staticmethod
    def activate(x):
        if abs(x) > 704:
            x = x / abs(x) * 704
        return 1 / (1 + math.exp(-x))

    def study(self, vectors, expected_values_vectors):
        while True:
            result_error = 0.0
            for q in range(len(expected_values_vectors)):
                vector = vectors[q]
                expected_values = expected_values_vectors[q]

                result = self.get_study_data(vector)
                errors = [{} for j in range(len(result) - 1)]
                for j in range(len(expected_values)):
                    errors[-1][j] = expected_values[j] - result[-1][j]
                    result_error += errors[-1][j] ** 2
                result_error /= 2.0

                for k in range(len(result) - 2):
                    for j in result[-k - 2]:
                        for p in range(len(self.layers[-k - 1][j].weights)):
                            if j not in errors[-k - 2]:
                                errors[-k - 2][j] = 0.0
                            errors[-k - 2][j] += errors[-k - 1][p] * self.layers[-k - 1][j].weights[p]

                for k in range(len(self.layers)):
                    for j in range(len(self.layers[k])):
                        for p in range(len(self.layers[k][j].weights)):
                            self.layers[k][j].weights[p] += self.learning_rate * result[k][j] * errors[k][p]

            if abs(result_error) < self.max_error:
                break

    def calculate(self, vector):
        return self.get_study_data(vector)[-1]

    def get_study_data(self, vector):
        data = [{}]
        for j in range(len(vector)):
            data[0][j] = vector[j]

        for p in range(len(self.layers)):
            new_vector_data = {}
            for j in range(len(self.layers[p])):
                node = self.layers[p][j]
                assert isinstance(node, Node)
                for k in range(len(node.weights)):
                    if k not in new_vector_data:
                        new_vector_data[k] = 0.0
                    new_vector_data[k] += node.calculate(k, data[p][j])
            for k in range(len(new_vector_data)):
                new_vector_data[k] = self.activate(new_vector_data[k])
            data.append(new_vector_data)
        return data


class Node:
    weights = None

    def __init__(self, weight_count):
        self.weights = []
        for i in range(weight_count):
            self.weights.append(random.random() * 0.2 + 0.1)

    def change_weight(self, idx, new_weight):
        self.weights[idx] = new_weight

    def calculate(self, idx, value):
        return value * self.weights[idx]


class ConvolutionalNetwork:
    perceptron = None
    original_map_size = None
    map_sizes = None

    def __init__(self, original_map_size, map_sizes, nodes_count_per_layer):
        self.perceptron = Perceptron(nodes_count_per_layer)
        self.original_map_size = original_map_size
        self.map_sizes = map_sizes

    @staticmethod
    def convolve(pattern, matrix_size, kernel_size):
        assert len(pattern) == matrix_size[1] + kernel_size[1] - 1
        assert len(pattern[0]) == matrix_size[2] + kernel_size[2] - 1

        # TODO kernel weights
        kernel = [[0.0 for i in range(kernel_size[1])] for j in range(kernel_size[2])]
        kernel[0][1] = 0.125
        kernel[1][0] = 0.125
        kernel[1][1] = 0.5
        kernel[1][2] = 0.125
        kernel[2][1] = 0.125
        new_matrix = [[0.0 for i in range(matrix_size[1])] for j in range(matrix_size[2])]
        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[0])):
                val = 0.0
                for k in range(len(kernel)):
                    for p in range(len(kernel[0])):
                        val += kernel[k][p] * pattern[i + k][j + p]
                new_matrix[i][j] = val
        return new_matrix

    def max_pool(self, pattern, matrix_size, kernel_size):
        assert len(pattern) == kernel_size[1] * matrix_size[1]
        assert len(pattern[0]) == kernel_size[2] * matrix_size[2]

        new_matrix = [[0 for i in range(matrix_size[1])] for j in range(matrix_size[2])]
        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[0])):
                val = -1E308
                for k in range(kernel_size[1]):
                    for p in range(kernel_size[2]):
                        val = max(val, pattern[2*i + k][2*j + p])
                new_matrix[i][j] = val
        return new_matrix

    def study(self, original_vectors, expected_values):
        result_original_vectors = []
        for j in range(len(original_vectors)):
            result_matrix = original_vectors[j]
            for i in range(self.map_sizes):
                convolved_matrix = self.convolve(result_matrix, self.map_sizes[i][1][1], self.map_sizes[i][1][2])
                result_matrix = self.max_pool(convolved_matrix, self.map_sizes[i][2][1], self.map_sizes[i][2][2])
            result_original_vectors.append(np.append([], result_matrix))
        self.perceptron.study(original_vectors, expected_values)

    def calculate(self, pattern):
        result_matrix = pattern
        for i in range(self.map_sizes):
            convolved_matrix = self.convolve(result_matrix, self.map_sizes[i][1][1], self.map_sizes[i][1][2])
            result_matrix = self.max_pool(convolved_matrix, self.map_sizes[i][2][1], self.map_sizes[i][2][2])
        return self.perceptron.calculate(np.append([], result_matrix))


network = ConvolutionalNetwork(
    (48, 48),
    ((((44, 44), (5, 5)), ((22, 22), (2, 2))), (((18, 18), (5, 5)), ((9, 9), (2, 2)))),
    [81, 70, 1]
)
