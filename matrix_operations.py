import math
import numpy as np
import copy


class MatrixOperations:
    @staticmethod
    def multiply_matrices(matrix1, matrix2):
        if len(matrix1[0]) != len(matrix2):
            raise Exception
        matrix1_x_axis, matrix1_y_axis = len(matrix1[0]), len(matrix1)
        matrix2_x_axis, matrix2_y_axis = len(matrix2[0]), len(matrix2)
        final_matrix = []
        for current_line_in_matrix1 in range(matrix1_y_axis):
            new_matrix_line = []
            for current_column_in_matrix2 in range(matrix2_x_axis):
                result_element = 0
                for current_column_in_matrix1 in range(matrix1_x_axis):
                    result_element += (matrix1[current_line_in_matrix1][current_column_in_matrix1] *
                                       matrix2[current_column_in_matrix1][current_column_in_matrix2])
                new_matrix_line.append(result_element)
            final_matrix.append(new_matrix_line)
        return final_matrix

    # line in matrix is a sum of products of multiplication of corresponding items of multiplied matrices

    @staticmethod
    def matrix_transpose(matrix0):
        transposed_matrix = [[matrix0[original_row][original_column] for original_row in range(len(matrix0))] for original_column in range(len(matrix0[0]))]
        return transposed_matrix

    # matrix transposition

    @staticmethod
    def vector_subtraction(vector1, vector2):
        if len(vector1) != len(vector2):
            print('Vectors have different size, subtraction is impossible')
            return -1
        result_vector = [vector1[index] - vector2[index] for index in range(len(vector1))]
        return result_vector

    # method for vector subtraction

    @staticmethod
    def matrix_subtraction(matrix1, matrix2):
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            print('Matrices have different size, subtraction is impossible')
            return -1
        subtracted_matrix = [[matrix1[matrix_rows][matrix_cols] - matrix2[matrix_rows][matrix_cols] for matrix_cols in range(len(matrix1[0]))] for matrix_rows in range(len(matrix1))]
        return subtracted_matrix

    # method for matrices subtraction

    @staticmethod
    def matrix_coefficient_multiplication(matrix, c):
        for rows in range(len(matrix)):
            for cols in range(len(matrix[0])):
                matrix[rows][cols] *= c
        return matrix

    # multiplication of matrix by coefficient

    @staticmethod
    def turn_pure_vector_into_shaped_image_matrix(list_of_pixel_rgb, rows, columns):
        list_of_pixel_rgb = [(list_of_pixel_rgb[index_of_current_neuron]+1)/2 for index_of_current_neuron in range(len(list_of_pixel_rgb))]
        return np.reshape(list_of_pixel_rgb, (rows, columns, 3))

    # turns pure vector into shaped image matrix

    @staticmethod
    def normalize_values_of_image_matrix(matrix):
        normalized_matrix = copy.deepcopy(matrix)
        for rows in range(len(normalized_matrix)):
            for columns in range(len(normalized_matrix[rows])):
                for colors in range(len(normalized_matrix[rows][columns])):
                    normalized_matrix[rows][columns][colors] = matrix[rows][columns][colors] * 2 - 1
        return normalized_matrix

    # function for normalization of values of shaped image matrix(fitting them in range from -1 to 1

    @staticmethod
    def mod_of_vector(vector):
        square_sum = 0
        for el in vector:
            square_sum += (el ** 2)
        return math.sqrt(square_sum)

    # function for finding vector's module
