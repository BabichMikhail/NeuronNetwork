import numpy as np
import matplotlib.image
import matplotlib.pyplot
import string


def activate(val):
    if val > 0.0:
        return 1
    return -1


def activate_matrix(matrix):
    mat = []
    for i in range(matrix.shape[0]):
        vec = []
        for j in range(matrix.shape[1]):
            vec.append(activate(matrix.item(i, j)))
        mat.append(vec)
    return np.matrix(mat)


def recognize(w, vec):
    y = np.transpose(np.matrix(vec))

    count = 0
    k = 195
    offset = 0
    y1 = None
    y2 = None
    while y2 is None or not np.array_equiv(y, y2):
        y2 = y1
        y1 = y
        y = activate_matrix(w * y)
        for i in range(k):
            y.itemset(((i + offset) % 225, 0), y1.item((i + offset) % 225, 0))
        offset += 30
        count += 1
    return np.transpose(y)


def get_letters(alphabet, directory_name):
    letters = []
    for char in alphabet:
        letter = []
        count = 0
        #print(matplotlib.image.imread("./" + directory_name + "/" + char + ".png"))
        for i in np.append([], matplotlib.image.imread("./" + directory_name + "/" + char + ".png")):
            letter.append(i)
            count += 1

        # if count == 900:
        #     print(char)
        #     print(letter)
        #     new_letter = []
        #     for i in range(225):
        #         val = 0
        #         for j in range(4):
        #             val += letter[j + 4 * i]
        #         new_letter.append(val / 4.0)
        #     letter = new_letter

        for i in range(225):
            val = letter[i]
            if val > 0.5:
                val = -1
            else:
                val = 1
            letter[i] = val

        # print(letter)
        letters.append(letter)
    return letters


def find_recognized_letter(letter, original_letters):
    idx = 0
    error_limit = 15
    for original_letter in original_letters:
        error_count = 0
        for i in range(225):
            if letter.item(0, i) != original_letter[i]:
                error_count += 1
        if error_count < error_limit:
            return idx
        idx += 1
    return -1


original_letters_set = string.ascii_lowercase[0:26]
noises_letters_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'k', 'l', 'q', 'z']

# original_letters = get_letters(['a', 'b', 'c', 'd'], 'selection')
original_letters = get_letters(original_letters_set, 'selection')
noises_letters = get_letters(noises_letters_set, 'tests')
W = None
for letter in original_letters:
    C = np.transpose(np.matrix(letter)) * np.matrix(letter)
    if W is None:
        W = C
    else:
        W = W + C


W = W - np.diag(np.diag(W))

for i in range(0, 225):
    pass
    #print(W[i])
# print(W[0])

print("Test network on original letters:")
index = 0
for letter in original_letters:
    # print(np.array_equiv(recognize(W, letter), letter))
    char_index = find_recognized_letter(recognize(W, letter), original_letters)
    if char_index == -1:
        print("error. letter not recognized")
    else:
        print("success. letter: " + original_letters_set[char_index] + ". expected: " + original_letters_set[index])
    index += 1

print()
print("Test network on noises letters:")
index = 0
for letter in noises_letters:
    # print(np.array_equiv(recognize(W, letter), original_letters[0]))
    char_index = find_recognized_letter(recognize(W, letter), original_letters)
    if char_index == -1:
        print("error. letter not recognized")
    else:
        print("success. letter: " + original_letters_set[char_index] + ". expected: " + noises_letters_set[index])
    index += 1
