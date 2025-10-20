import numpy as np
import warnings


class AHP:
    def __init__(self, criteria):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria
        self.num_criteria = criteria.shape[0]

    def calculate_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, "the matrix is not orthogonal"
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError("the matrix is not symmetric")
        eigen_values, eigen_vectors = np.linalg.eig(input_matrix)
        max_eigen = np.max(eigen_values)
        max_index = np.argmax(eigen_values)
        eigen = eigen_vectors[:, max_index]
        eigen = eigen / eigen.sum()
        if n > 9:
            CR = None
            warnings.warn("can not judge the uniformity")
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n - 1]
        return max_eigen, CR, eigen

    def calculate_mean_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, "the matrix is not orthogonal"
        A_mean = []
        for i in range(n):
            mean_value = input_matrix[:, i] / np.sum(input_matrix[:, i])
            A_mean.append(mean_value)
        eigen = []
        A_mean = np.array(A_mean)
        for i in range(n):
            eigen.append(np.sum(A_mean[:, i]) / n)
        eigen = np.array(eigen)
        matrix_sum = np.dot(input_matrix, eigen)
        max_eigen = np.mean(matrix_sum / eigen)
        if n > 9:
            CR = None
            warnings.warn("can not judge the uniformity")
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n - 1]
        return max_eigen, CR, eigen

    def run(self, method="calculate_weights"):
        weight_func = eval(f"self.{method}")
        max_eigen, CR, criteria_eigen = weight_func(self.criteria)
        print('准则层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(max_eigen, CR, '' if CR < 0.1 else '不'))
        print('准则层权重={}\n'.format(criteria_eigen))
        return criteria_eigen
