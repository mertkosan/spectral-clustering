from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
import numpy as np

import time
import json


def laplacian(A):
    return np.diag(A.sum(axis=0)) - A


def normalized_laplacian(A):
    L = laplacian(A)
    D = np.diag(1 / np.diagonal(L))
    return np.sqrt(D) @ L @ np.sqrt(D)


def first_k_eigenvectors_sparse(M, k, which='SM'):
    return eigsh(M, k, which=which)[1]


def first_k_eigenvectors(M, k):
    eig_val, eig_vec = eigh(M)
    return np.fliplr(eig_vec[:, :k])


def spectral_embeddings(matrix, k, measure_time=False, not_sparse=False):
    if not_sparse:
        sparse_length = 10000000000000000
    else:
        start_sparse = time.time()
        try:
            eigv_sparse = first_k_eigenvectors_sparse(matrix, k)
            end_sparse = time.time()
        except Exception as e:
            end_sparse = time.time() + 10000000000000000
            print(e)
        sparse_length = end_sparse - start_sparse

    matrix_dense = matrix.todense()
    start_full = time.time()
    eigv_full = first_k_eigenvectors(matrix_dense, k)
    end_full = time.time()
    full_length = end_full - start_full

    if measure_time:
        return (eigv_sparse, sparse_length, 'Sparse') if sparse_length < full_length else (eigv_full, full_length, 'Full')
    else:
        return eigv_sparse if sparse_length < full_length else eigv_full


def save_json(obj, path):
    with open(path, 'w') as _file:
        json.dump(obj, _file)


def load_json(path):
    with open(path) as _file:
        return json.load(_file)
