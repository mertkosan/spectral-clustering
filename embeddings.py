import numpy as np
import scipy.sparse as sp

import datasets
import utils

import argparse

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', help='Datasets: cora, email, ssets')
parser.add_argument('--version', default='1', help='version for ssets, default 1 for others')
args = parser.parse_args()

if __name__ == '__main__':
    A, labels = datasets.load_graph(args.dataset, args.version)

    # dense
    if not isinstance(A, np.ndarray):
        A = np.array(A.todense())
    L = utils.laplacian(A)
    N = utils.normalized_laplacian(A)

    # sparse
    A = sp.csr_matrix(A)
    L = sp.csr_matrix(L)
    N = sp.csr_matrix(N)

    matrices = {
        'A': A,
        'L': L,
        'N': N
    }

    for matrix_id in matrices:
        matrix = matrices[matrix_id]
        eig_val, eig_vec = np.linalg.eigh(matrix.todense())
        path = f"{args.dataset}/embeddings/{matrix_id}_{args.dataset}_v{args.version}.npy"
        np.save(path, eig_vec)
