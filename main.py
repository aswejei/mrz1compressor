import matplotlib.image as image
import matplotlib.pyplot as plot
import math
import numpy as np
import random
import ast

from matrix_operations import MatrixOperations as mo

img_matrix = image.imread('midicat1.png')
pixels_per_block = 16  # 4x4
amount_of_hidden_neurons = 4 * 3
e = 2500
threshold_error = 0

print("Pick work mode:\n1 - teach weights matrix\nAnything else - use weights matrix")
mode = input()
decision = '0'

if mode == '1':
    img_name = input("Enter name of file with target picture: ")
    img_matrix = image.imread(img_name)
    pixels_per_block = int(input("Enter amount of pixels in one block: "))
    amount_of_hidden_neurons = int(input("Enter amount of neurons on the hidden layer: "))
    e = int(input("Enter threshold value of error: "))
    threshold_error = 0
else:
    weights_matrix_file = input("Enter name of the file with matrices of weight coefficients: ")
    decision = input("Enter \"1\" if you want to compress picture into archive; \nEnter whatever else if you want to "
                     "decompress picture from the archive: ")
    transformation_file_name = input("Enter name of the file/image, you would like to transform: ")
    if weights_matrix_file == '' or transformation_file_name == '':
        print("Data wasn\'t provided, exiting...")
        exit()
    weight_matrices_file = open(weights_matrix_file, "r")

    weight_matrix1 = ast.literal_eval(weight_matrices_file.readline())
    weight_matrix2 = ast.literal_eval(weight_matrices_file.readline())
    pixels_per_block = len(weight_matrix2[0]) / 3

    if decision != '1':
        archive = open(transformation_file_name, "r")
        compressed_image = ast.literal_eval(archive.readline())
        image_vector_of_rgbs_renewed = list()
        for block in range(len(compressed_image)):
            renewed_matrix = mo.multiply_matrices([compressed_image[block]], weight_matrix2)
            renewed_matrix[0] = list(map(lambda x: (x + 1) / 2, renewed_matrix[0]))
            image_vector_of_rgbs_renewed += renewed_matrix
            if int(block / len(compressed_image) * 100) % 10 == 0 and block % 100 == 0:
                print(int(block / len(compressed_image) * 100), "%")

        renewed_vector_of_rgbs_ready = []
        result_compression = (math.sqrt((pixels_per_block * 3) / amount_of_hidden_neurons))
        block_size = int(math.sqrt(pixels_per_block))

        size_of_block_compressed = int(math.sqrt(amount_of_hidden_neurons / 3))
        row_blocks_amount = img_matrix.shape[1] // block_size
        column_blocks_amount = img_matrix.shape[0] // block_size

        for block_row in range(column_blocks_amount):
            current_starting_block = block_row * row_blocks_amount
            for block_index_in_row in range(int(block_size)):
                for current_block in image_vector_of_rgbs_renewed[
                                     current_starting_block:current_starting_block + row_blocks_amount]:
                    renewed_vector_of_rgbs_ready.append(current_block[(3 * block_index_in_row * block_size):
                                                                      (3 * (block_index_in_row + 1) * block_size)])

        new_image = mo.turn_pure_vector_into_shaped_image_matrix(list(np.array(renewed_vector_of_rgbs_ready).flatten()),
                                                                 img_matrix.shape[0], img_matrix.shape[1])
        plot.imshow(new_image)
        plot.savefig(transformation_file_name[:-4] + ".png")
        exit()

    img_matrix = image.imread(transformation_file_name)

block_size = int(math.sqrt(pixels_per_block))
block_matrix = []

img_matrix = img_matrix[:, :, :3]  # if we have alpha value, we ignore it(shorten by 3 values)

leftover_lines = len(img_matrix) % math.sqrt(pixels_per_block)
lines_to_complete_shape = math.sqrt(pixels_per_block) - leftover_lines

leftover_columns = len(img_matrix[0]) % math.sqrt(pixels_per_block)
columns_to_complete_shape = math.sqrt(pixels_per_block) - leftover_columns

WHITE = [1., 1., 1.]
if leftover_lines != 0:  # append lines of white pixels
    blank_line = []
    for index_of_column in range(len(img_matrix[0])):
        blank_line.append(WHITE)
    blank_space = [blank_line]
    blank_space *= int(lines_to_complete_shape)
    img_matrix = np.append(img_matrix, blank_space, 0)

if leftover_columns != 0:  # append rows of white pixels
    blank_column = []
    for index_of_row in range(len(img_matrix)):
        blank_column.append([WHITE])
    for index_of_column in range(int(columns_to_complete_shape)):
        img_matrix = np.append(img_matrix, blank_column, 1)

shape = img_matrix.shape
number_of_neurons_layer1 = shape[0] * shape[1] * shape[2]  # vertical, horizontal, colour
number_of_neurons_layer2 = number_of_neurons_layer1 / pixels_per_block

neuron_matrix = mo.normalize_values_of_image_matrix(img_matrix)

for i in range(0, len(neuron_matrix), block_size):
    for j in range(0, len(neuron_matrix[i]), block_size):
        block_matrix.append([])
        for k in range(i, i + block_size):
            block_matrix[-1].extend(neuron_matrix[k][j:j + block_size])
# block separation

compressed_image_vector_of_rgbs = list()
renewed_image_vector_of_rgbs = list()

if threshold_error == 0:
    threshold_error = len(block_matrix)

# ---------------------------------------- обучение сети -----------------------------------------------------

if mode == '1':
    print("Preparations complete!")

    iterations = 0

    # ------- weights matrices preparation ---------
    total_quadratic_error = e + 1

    weight_matrix1 = []
    for row in range(pixels_per_block * 3):
        weight_matrix1.append([])
        for column in range(amount_of_hidden_neurons):
            random_weight = ((random.random() * 2) - 1) / 1000
            weight_matrix1[row].append(random_weight)
    weight_matrix2 = mo.matrix_transpose(weight_matrix1)

    while total_quadratic_error > e:
        total_quadratic_error = 0
        iterations += 1
        for learning_sample_index in range(threshold_error):
            # running through all the blocks and changing weights until we get "good" error everywhere
            input_layer = [list(np.array(block_matrix[learning_sample_index]).flatten())]
            input_layer_number = len(input_layer[0])

            compressed_matrix = list(np.matmul(np.array(input_layer), np.array(weight_matrix1)))
            renewed_matrix = list(np.matmul(np.array(compressed_matrix), np.array(weight_matrix2)))
            difference_vector = mo.vector_subtraction(renewed_matrix[0], input_layer[0])

            input_layer_transposed = np.array(input_layer).T
            result_matrix_transposed = np.array(compressed_matrix).T

            # ------------------------ calculation of learning coefficient ------------------------

            # alpha_weight_matrix1 = 1 / (np.matmul(np.array(input_layer), input_layer_transposed)[0][0])
            # alpha_weight_matrix2 = 1 / (np.matmul(np.array(compressed_matrix), result_matrix_transposed)[0][0])
            # Alternative(using it, because using self-adjusting
            # learning step led to completely inaccurate value deltas) #
            alpha_weight_matrix1 = alpha_weight_matrix2 = 0.0005

            # --------- correcting weight matrices ---------

            XiT_dXi = np.matmul(input_layer_transposed, np.array([difference_vector]))
            XiT_dXi_W2T = np.matmul(XiT_dXi, np.array(weight_matrix2).T)
            weight_matrix1 = mo.matrix_subtraction(weight_matrix1, list(alpha_weight_matrix1 * XiT_dXi_W2T))

            YiT_dXi = np.matmul(result_matrix_transposed, np.array([difference_vector]))
            weight_matrix2 = mo.matrix_subtraction(weight_matrix2, list(alpha_weight_matrix2 * YiT_dXi))

            # ----------- normalization of weight matrices -----------

            weight_matrix1_transposed = mo.matrix_transpose(weight_matrix1)
            for weight_matrix1_row in range(len(weight_matrix1)):
                for weight_matrix1_column in range(len(weight_matrix1[weight_matrix1_row])):
                    denominator1 = mo.mod_of_vector(weight_matrix1_transposed[weight_matrix1_column])
                    weight_matrix1[weight_matrix1_row][weight_matrix1_column] /= denominator1

            # ------ normalization of weight matrix No. 1 ------

            weight_matrix2_transposed = mo.matrix_transpose(weight_matrix2)
            for weight_matrix2_row in range(len(weight_matrix2)):
                for weight_matrix2_column in range(len(weight_matrix2[weight_matrix2_row])):
                    denominator2 = mo.mod_of_vector(weight_matrix2_transposed[weight_matrix2_column])
                    weight_matrix2[weight_matrix2_row][weight_matrix2_column] /= denominator2
            # ------ normalization of weight matrix No. 2 ------

            # ---------- подсчёт ошибки ----------

            epoch_quadratic_error = 0
            for i in range(input_layer_number):
                epoch_quadratic_error += (difference_vector[i] ** 2)
            total_quadratic_error += epoch_quadratic_error

        print("Iteration " + str(iterations) + ": E = " + str(total_quadratic_error) + "; e = " + str(e))

for block_index in range(len(block_matrix)):
    layer1 = [list(np.array(block_matrix[block_index]).flatten())]
    layer1_number = len(layer1[0])

    compressed_matrix = mo.multiply_matrices(layer1, weight_matrix1)
    renewed_matrix = mo.multiply_matrices(compressed_matrix, weight_matrix2)

    for x in renewed_matrix[0]:
        x += 1
        x /= 2

    # image recreation

    compressed_image_vector_of_rgbs += compressed_matrix
    renewed_image_vector_of_rgbs += renewed_matrix

    if int(block_index / len(block_matrix) * 100) % 10 == 0 and block_index % 100 == 0:
        print(int(block_index / len(block_matrix) * 100), "%")

compressed_vector_of_rgbs_ready = []
renewed_vector_of_rgbs_ready = []
result_compression = (math.sqrt((pixels_per_block * 3) / amount_of_hidden_neurons))

compressed_block_size = int(math.sqrt(amount_of_hidden_neurons / 3))
row_blocks_amount = img_matrix.shape[1] // block_size
column_blocks_amount = img_matrix.shape[0] // block_size

for block_row in range(column_blocks_amount):
    current_starting_block = block_row * row_blocks_amount
    for block_index_in_row in range(int(block_size)):
        for current_block in renewed_image_vector_of_rgbs[current_starting_block: \
                current_starting_block + row_blocks_amount]:
            renewed_vector_of_rgbs_ready.append(current_block[(3 * block_index_in_row * block_size):
                                                              (3 * (block_index_in_row + 1) * block_size)])

for x in compressed_image_vector_of_rgbs:
    for i in range(compressed_block_size):
        compressed_vector_of_rgbs_ready.append(x[(3 * i * compressed_block_size):
                                                 ((i + 1) * 3 * compressed_block_size)])

plot.imshow(mo.turn_pure_vector_into_shaped_image_matrix(list(np.array(compressed_vector_of_rgbs_ready).flatten()),
                                                         int(img_matrix.shape[0] / result_compression),
                                                         int(img_matrix.shape[1] / result_compression)))
plot.show()
# compressed image

if decision == '1':
    name_of_archive = input("Enter archive name: ")
    new_archive = open(name_of_archive, "w")
    new_archive.write(str(compressed_image_vector_of_rgbs))

if decision != '1':
    plot.imshow(mo.turn_pure_vector_into_shaped_image_matrix(list(np.array(renewed_vector_of_rgbs_ready).flatten()),
                                                             img_matrix.shape[0], img_matrix.shape[1]))
plot.show()
# decompressed image

if mode == '1':
    decision_to_save_matrices = input("Press enter if you don\'t want to save weights matrix; enter any other value "
                                      "to save matrix in corresponding file: ")
    if decision_to_save_matrices != '':
        if not decision_to_save_matrices.endswith(".txt"):
            decision_to_save_matrices = decision_to_save_matrices + ".txt"
        weight_matrices_file = open(decision_to_save_matrices, 'w')
        weight_matrices_file.write(str(weight_matrix1) + "\n" + str(weight_matrix2))
        weight_matrices_file.close()

print("End of program!")
