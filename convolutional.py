import numpy as np
import math
import random
import matplotlib.image
import matplotlib.pyplot
import time
import glob
import string


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
                print(result_error)
                print(result)
                print(expected_values)
                print(errors)

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

    learning_rate = 0.1
    max_error = 0.1

    convolve_weights = None
    full_connection_weights = None

    def __init__(self, original_map_size, convolve_weight_counts, fc_weight_counts):
        self.perceptron = Perceptron(fc_weight_counts)
        self.original_map_size = original_map_size
        self.map_sizes = convolve_weight_counts

        self.convolve_weights = []
        for p in range(len(convolve_weight_counts)):
            level_weights = []
            for i in range(3):
                kernel_weights = []
                size = convolve_weight_counts[p][0] * convolve_weight_counts[p][1]
                sum = 0.0
                for j in range(convolve_weight_counts[p][0]):
                    weights = []
                    for k in range(convolve_weight_counts[p][1]):
                        weight = random.random() * 0.2 + 0.1
                        sum += weight
                        weights.append(weight)
                    kernel_weights.append(weights)

                coef = sum / size
                for j in range(len(kernel_weights)):
                    for k in range(len(kernel_weights[j])):
                        kernel_weights[j][k] /= coef

                level_weights.append(kernel_weights)
            self.convolve_weights.append(level_weights)
        # print(self.convolve_weights)

        self.full_connection_weights = []
        for i in range(len(fc_weight_counts) - 1):
            node_weights = []
            for j in range(fc_weight_counts[i]):
                weights = []
                for k in range(fc_weight_counts[i + 1]):
                    weights.append(random.random() * 0.2 + 0.1)
                node_weights.append(weights)
            self.full_connection_weights.append(node_weights)

    @staticmethod
    def activate(x):
        if abs(x) > 704:
            x = x / abs(x) * 704
        return 1 / (1 + math.exp(-x))

    def study(self, original_mats, expected_values_vectors):
        while True:
            result_error = 0.0
            for q in range(len(expected_values_vectors)):
                expected_values = expected_values_vectors[q]

                convolve_result, result = self.get_study_data(original_mats[q])
                full_conn_errors = [{} for j in range(len(result) - 1)]
                for j in range(len(expected_values)):
                    full_conn_errors[-1][j] = expected_values[j] - result[-1][j]
                    result_error += full_conn_errors[-1][j] ** 2
                print(result_error)
                print(len(result))
                print(expected_values)
                print(full_conn_errors)

                for k in range(len(result) - 2):
                    for j in result[-k - 2]:
                        for p in range(len(self.full_connection_weights[-k - 1][j])):
                            if j not in full_conn_errors[-k - 2]:
                                full_conn_errors[-k - 2][j] = 0.0
                            full_conn_errors[-k - 2][j] += full_conn_errors[-k - 1][p] * self.full_connection_weights[-k - 1][j][p]
                    # print(full_conn_errors[-k - 2])

                for k in range(len(self.full_connection_weights)):
                    for j in range(len(self.full_connection_weights[k])):
                        for p in range(len(self.full_connection_weights[k][j])):
                            self.full_connection_weights[k][j][p] += self.learning_rate * result[k][j] * full_conn_errors[k][p]

                convolve_errors = [{} for j in range(len(convolve_result) - 1)]
                convolve_errors[-1] = full_conn_errors[0]
                # convolve_error[-1] = full_conn_errors[0]
                for k in range(len(convolve_result) - 2):
                    for i in range(3):
                        left = i * len(result)
                        right = (i + 1) * len(result)
                        convolve_error_part = convolve_errors[-1][left:right]
                        for j in range(len(convolve_error_part)):
                            for p in range(len(convolve_result[k][1][j])):
                                for q in range(len(convolve_result[k][1][j][p])):
                                    # convolve_errors[-k - 2]
                                    # convolve_result[k][1]
                                    pass

            if abs(result_error) < self.max_error:
                break

    def calculate(self, pattern):
        return self.get_study_data(pattern)[-1]

    def get_study_data(self, pattern):
        convolve_data = [(pattern, pattern)]
        for p in range(len(self.convolve_weights)):
            new_layers_data = []
            for q in range(len(convolve_data[p][1])):
                layer = []
                # if q == 0:
                #     print('abc')
                #     print(len(convolve_data[p][1][q]))
                #     print(len(convolve_data[p][1][q]) - len(self.convolve_weights[p]) + 1)
                for w in range(len(convolve_data[p][1][q]) - len(self.convolve_weights[p][q]) + 1):
                    line = []
                    # if w == 0: print(len(convolve_data[p][1][q][w]) - len(self.convolve_weights[p][0]) + 1)
                    for e in range(len(convolve_data[p][1][q][w]) - len(self.convolve_weights[p][q][0]) + 1):
                        value = 0.0
                        for j in range(len(self.convolve_weights[p][q])):
                            for k in range(len(self.convolve_weights[p][q][j])):
                                # print(convolve_data[p][1][q][w + j][e + k])
                                # print(self.convolve_weights[p][q][j][k])
                                value += convolve_data[p][1][q][w + j][e + k] * self.convolve_weights[p][q][j][k]
                        line.append(value)
                    # print(line)
                    layer.append(line)
                # print(layer)
                new_layers_data.append(layer)

            # print(new_layers_data)
            new_layer_max_pool_data = []
            for q in range(len(new_layers_data)):
                layer = []
                # if q == 0: print(len(new_layers_data))
                for w in range(len(new_layers_data[q]) // 2):
                    # if w == 0: print(len(new_layers_data[q]))
                    line = []
                    for e in range(len(new_layers_data[q][w]) // 2):
                        # if e == 0: print(len(new_layers_data[q][w]))
                        value = -1E308
                        for j in range(2):
                            for k in range(2):
                                value = max(value, new_layers_data[q][2 * w + j][2 * e + k])
                        line.append(value)
                    layer.append(line)
                new_layer_max_pool_data.append(layer)
            # print(new_layer_max_pool_data)
            convolve_data.append((new_layers_data, new_layer_max_pool_data))

        last_convolve_data = convolve_data[-1][1]
        vector_data = []
        for p in range(len(last_convolve_data)):
            # if p == 0: print(len(last_convolve_data))
            for j in range(len(last_convolve_data[p])):
                # if j == 0: print(len(last_convolve_data[p]))
                for k in range(len(last_convolve_data[p][j])):
                    # if k == 0: print(len(last_convolve_data[p][j]))
                    vector_data.append(last_convolve_data[p][j][k])

        data = [{}]
        for j in range(len(vector_data)):
            data[0][j] = vector_data[j]

        print(len(data[0]))
        print("hello")

        for p in range(len(self.full_connection_weights)):
            new_vector_data = {}
            for j in range(len(self.full_connection_weights[p])):
                for k in range(len(self.full_connection_weights[p][j])):
                    if k not in new_vector_data:
                        new_vector_data[k] = 0.0
                    new_vector_data[k] += self.full_connection_weights[p][j][k] * data[p][j]
            for k in range(len(new_vector_data)):
                new_vector_data[k] = self.activate(new_vector_data[k])
            data.append(new_vector_data)
        return convolve_data, data

    @staticmethod
    def convolve(pattern, matrix_size, kernel_size):
        print(len(pattern))
        print(matrix_size)
        print(kernel_size)
        assert len(pattern) == matrix_size[0] + kernel_size[0] - 1
        assert len(pattern[0]) == matrix_size[1] + kernel_size[1] - 1

        # TODO kernel weights
        kernel = [[0.0 for i in range(kernel_size[0])] for j in range(kernel_size[1])]
        kernel[0][1] = 0.125
        kernel[1][0] = 0.125
        kernel[1][1] = 0.5
        kernel[1][2] = 0.125
        kernel[2][1] = 0.125
        new_matrix = [[0.0 for i in range(matrix_size[0])] for j in range(matrix_size[1])]
        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[0])):
                val = 0.0
                for k in range(len(kernel)):
                    for p in range(len(kernel[0])):
                        val += kernel[k][p] * pattern[i + k][j + p]
                new_matrix[i][j] = val
        return new_matrix

    @staticmethod
    def max_pool(pattern, matrix_size, kernel_size):
        print(len(pattern))
        print(kernel_size)
        print(matrix_size)
        assert abs(len(pattern) - kernel_size[0] * matrix_size[0]) < 2
        assert abs(len(pattern[0]) - kernel_size[1] * matrix_size[1]) < 2

        new_matrix = [[0 for i in range(matrix_size[0])] for j in range(matrix_size[1])]
        for i in range(len(new_matrix) // 2 * 2):
            for j in range(len(new_matrix[0]) // 2 * 2):
                val = -1E308
                for k in range(kernel_size[0]):
                    for p in range(kernel_size[1]):
                        val = max(val, pattern[2*i + k][2*j + p])
                new_matrix[i][j] = val
        return new_matrix

    def study1(self, original_mats, expected_values):
        result_original_vectors = []
        for j in range(len(original_mats)):
            result_vector = []
            for k in range(len(original_mats[j])):
                result_matrix = original_mats[j][k]
                for i in range(len(self.map_sizes)):
                    convolved_matrix = self.convolve(result_matrix, self.map_sizes[i][0][0], self.map_sizes[i][0][1])
                    result_matrix = self.max_pool(convolved_matrix, self.map_sizes[i][1][0], self.map_sizes[i][1][1])
                result_vector.append(result_matrix)
            result_original_vectors.append(np.append([], result_vector))
        self.perceptron.study(result_original_vectors, expected_values)

    # def calculate(self, pattern):
    #     vector = []
    #     for k in range(len(pattern)):
    #         result_matrix = pattern[k]
    #         for i in range(len(self.map_sizes)):
    #             convolved_matrix = self.convolve(result_matrix, self.map_sizes[i][0][0], self.map_sizes[i][0][1])
    #             result_matrix = self.max_pool(convolved_matrix, self.map_sizes[i][1][0], self.map_sizes[i][1][1])
    #         vector.append(np.append([], result_matrix))
    #     return self.perceptron.calculate(np.append([], vector))


# network = ConvolutionalNetwork(
#     (250, 250),
#     ((((246, 246), (5, 5)), ((123, 123), (2, 2))), (((119, 119), (5, 5)), ((59, 59), (2, 2))), (((55, 55), (5, 5)), ((27, 27), (2, 2))), (((23, 23), (5, 5)), ((11, 11), (2, 2)))),
#     [363, 140, 1]
# )

network = ConvolutionalNetwork(
    [250, 250],
    # 240x240 120x120 / 116x116 58x58 / 54x54 27x27 / 24x24 12x12
    [[11, 11], [5, 5], [5, 5], [4, 4]],  # 12x12
    [12*12*3, 140, 1]
)

# fnames = [
#     "./faces/Aaron_Eckhart/Aaron_Eckhart_0001.jpg",
#     "./faces/Aaron_Guiel/Aaron_Guiel_0001.jpg"
#     "./faces_tests/1.jpg",
# ]


def get_file_rbg_mats(image):
    color_count = 3
    rgb_mats = [[] for t in range(color_count)]
    for image_line in image:
        rgb_pixels = [[] for t in range(color_count)]
        for image_pixel in image_line:
            for k in range(color_count):
                rgb_pixels[k].append(image_pixel[k] / 255.0)
        for k in range(color_count):
            rgb_mats[k].append(rgb_pixels[k])
    return rgb_mats


faces_path = "./faces/*/*.jpg"
faces_fnames = glob.glob(faces_path)[:10]
print(len(faces_fnames))
# string

not_faces_path = "./not_faces/*.jpg"
not_faces_fnames = glob.glob(not_faces_path)

random.seed(199)

faces_mats = []
faces_answers = []

print(len(not_faces_fnames))
not_faces_fnames.append("./faces_tests/1.jpg")
print(len(faces_fnames))

for fname in faces_fnames:
    start_time = time.time()
    rgb_mats = get_file_rbg_mats(matplotlib.image.imread(fname))
    # print(rgb_mats)
    faces_mats.append(rgb_mats)
    print(time.time() - start_time)
    faces_answers.append([1])
    # print(matplotlib.image.imread(fname))
    # print(len(matplotlib.image.imread(fname)))
    # matplotlib.pyplot.imread()
    # img_vector = np.append([], matplotlib.image.imread(fname))

for fname in not_faces_fnames:
    start_time = time.time()
    rgb_mats = get_file_rbg_mats(matplotlib.image.imread(fname))
    # print(rgb_mats)
    faces_mats.append(rgb_mats)
    print(time.time() - start_time)
    faces_answers.append([0])
    # print(matplotlib.image.imread(fname))
    # print(len(matplotlib.image.imread(fname)))
    # matplotlib.pyplot.imread()
    # img_vector = np.append([], matplotlib.image.imread(fname))

network.study(faces_mats, faces_answers)

print(network.calculate(get_file_rbg_mats(matplotlib.image.imread("./faces_tests/1.jpg"))))