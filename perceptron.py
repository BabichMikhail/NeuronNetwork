import numpy as np
import matplotlib.image
import math
import random
import string
import time


def activate(x):
    if abs(x) > 704:
        x = x / abs(x) * 704
    # if x == 0:
    #     print("error")
    #     return 0
    # print(x)
    # print(math.exp(-x))
    return 1 / (1 + math.exp(-x))


def activate_2(x):
    if abs(x) > 0.5:
        return x / abs(x)
    return 0


class Network:
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

    def study(self, vectors, expected_values_vectors):
        # self.layers[0][0].weights[0] = 0.24231929306082867
        # self.layers[0][0].weights[1] = 0.22358895795880857
        # self.layers[0][1].weights[0] = 0.23771243995545027
        # self.layers[0][1].weights[1] = 0.17955784671112549
        # self.layers[1][0].weights[0] = 0.11033213185736296
        # self.layers[1][1].weights[0] = 0.25490633335020907

        # self.layers[0][0].weights[0] = 0.3967842776611369
        # self.layers[0][0].weights[1] = 0.610457824649101
        # self.layers[0][1].weights[0] = 0.3811442113700222
        # self.layers[0][1].weights[1] = 0.5409360800663972
        # self.layers[1][0].weights[0] = 0.11033213185736296
        # self.layers[1][1].weights[0] = 0.554906333350209

        result_error = self.max_error + 1.0
        it = 0
        while True:
            result_error = 0.0
            for q in range(len(expected_values_vectors)):
                vector = vectors[q]
                expected_values = expected_values_vectors[q]

                result = self.get_study_data(vector)
                # print(vector)
                # print(expected_values)
                # print(result[-1])

                errors = [{} for j in range(len(result) - 1)]
                for j in range(len(expected_values)):
                    errors[-1][j] = expected_values[j] - result[-1][j]
                    result_error += errors[-1][j] ** 2
                result_error /= 2.0
                # print(result_error)
                # print(errors)

                for k in range(len(result) - 2):
                    for j in result[-k - 2]:
                        # for p in errors[-k - 1]:
                        for p in range(len(self.layers[-k - 1][j].weights)):
                            if j not in errors[-k - 2]:
                                errors[-k - 2][j] = 0.0
                            errors[-k - 2][j] += errors[-k - 1][p] * self.layers[-k - 1][j].weights[p]

                # print("Data")
                # print(result_error)
                # print(errors)
                # print(result)
                # print("qwerty")
                # print(vector)
                for k in range(len(self.layers)):
                    for j in range(len(self.layers[k])):
                        for p in range(len(self.layers[k][j].weights)):
                            # print(self.learning_rate * errors[k + 1][p])
                            # if k == 1:
                            #     print(result[k][j])
                            #     print(errors[k][p])
                            self.layers[k][j].weights[p] += self.learning_rate * result[k][j] * errors[k][p]
                        # print(self.layers[k][j].weights)

                # print("results: " + str(it))
                # it += 1
                # for i in range(len(self.layers)):
                #     for j in range(len(self.layers[i])):
                #         for k in range(len(self.layers[i][j].weights)):
                #             print(self.layers[i][j].weights[k])
                # for i in range(2):
                #     print(result[-2][i])
            print(result_error)

            if abs(result_error) < self.max_error:
                break

    def calculate(self, vector):
        return self.get_study_data(vector)[-1]

    def get_study_data(self, vector):
        data = [{}]
        for j in range(len(vector)):
            data[0][j] = vector[j]

        # print("study")
        for p in range(len(self.layers)):
            new_vector_data = {}
            for j in range(len(self.layers[p])):
                node = self.layers[p][j]
                assert isinstance(node, Node)
                for k in range(len(node.weights)):
                    if k not in new_vector_data:
                        new_vector_data[k] = 0.0
                    # if p == 0:
                    #     print(data[p][j])
                    #     print(node.weights[k])
                    new_vector_data[k] += node.calculate(k, data[p][j])
            # if p == 0:
            #     print(new_vector_data)
            for k in range(len(new_vector_data)):
                new_vector_data[k] = activate(new_vector_data[k])
                # if abs(new_vector_data[k]) > 0.5:
                #     new_vector_data[k] = new_vector_data[k] / abs(new_vector_data[k])
                # else:
                #     new_vector_data[k] = 0.0
            data.append(new_vector_data)
        return data


class Node:
    weights = None

    def __init__(self, weight_count):
        self.weights = []
        for i in range(weight_count):
            self.weights.append(random.random() * 0.2 + 0.1)
            # self.weights.append(random.random() / weight_count)

    def change_weight(self, idx, new_weight):
        self.weights[idx] = new_weight

    def calculate(self, idx, value):
        return value * self.weights[idx]


def get_letters(alphabet, directory_name):
    letters = []
    for char in alphabet:
        letter = []
        count = 0
        for j in np.append([], matplotlib.image.imread("./" + directory_name + "/" + char + ".png")):
            letter.append(j)
            count += 1

        for j in range(225):
            val = letter[j]
            # if val > 0.5:
            #     val = 1
            # else:
            #     val = -1

            letter[j] = math.tanh(1 - val)

        letters.append(letter)
    return letters


def idx_to_value_vector(idx):
    result = []
    k = 1
    for j in range(5):
        result.append(((idx + 1) & k) // k)
        k *= 2
    return result


def find_letter(result, results):
    min_error = 100
    idx = -1

    k = 1
    for i in range(len(results)):
        error = 0.0
        for j in range(len(results[i])):
            error += (results[i][j] - result[j]) ** 2
        if error < min_error:
            min_error = error
            idx = i
    # for i in range(5):
    #     k *= 2
    #
    # if 27.5 < sum < 0.5:
    #     return -1
    print(min_error)
    # idx = round(sum)
    return idx


original_letters_set = string.ascii_lowercase
noises_letters_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'k', 'l', 'q', 'z']

original_letters = get_letters(original_letters_set, 'selection')
noises_letters = get_letters(noises_letters_set, 'tests')

# random.seed(time.time().)
random.seed(199)

# network = Network([2, 2, 1])
# network.study([
#     [0, 0],
#     [1, 0],
#     [0, 1],
#     [1, 1],
# ], [
#     [0], [1], [1], [0]
# ])
#
# print(network.calculate([1, 1]))
# print(network.calculate([0, 1]))
# print(network.calculate([1, 0]))
# print(network.calculate([0, 0]))

start_time = time.time()

network = Network([len(original_letters[0]), 128, 5])

vectors = []
results = []
for i in range(len(original_letters)):
    results.append(idx_to_value_vector(i))
    # results.append([i])
    vectors.append(original_letters[i])
network.study(vectors, results)

print("--- %s seconds ---" % (time.time() - start_time))

for i in range(len(original_letters)):
    idx = find_letter(network.calculate(original_letters[i]), results)
    if idx == -1:
        print("Letter not recognized")
    else:
        print("Expected letter: " + original_letters_set[i] + ". Recognized letter: " + original_letters_set[idx])

print()
for i in range(len(noises_letters)):
    idx = find_letter(network.calculate(noises_letters[i]), results)
    if idx == -1:
        print("Letter not recognized")
    else:
        print("Expected letter: " + noises_letters_set[i] + ". Recognized letter: " + original_letters_set[idx])
