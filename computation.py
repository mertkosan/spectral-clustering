import networkx as nx
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt

import datasets
import utils
import learning

import matplotlib.pyplot as plt

import argparse

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', help='Datasets: cora, email, ssets')
parser.add_argument('--version', default='1', help='version for ssets (1,2,3,4), default 1 for others')
parser.add_argument('--plot', action='store_true', help='already have results, want to plot.')
args = parser.parse_args()

if __name__ == '__main__':
    if not args.plot:
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

        time = {'A': [], 'L': [], 'N': []}
        for matrix_id in matrices:
            matrix = matrices[matrix_id]
            print(matrix_id)
            if matrix_id == 'A':
                eigv, time_elapsed, desp = utils.spectral_embeddings(matrix, 20, measure_time=True, not_sparse=matrix_id == 'A')
                time[matrix_id].append(time_elapsed)
                print(time_elapsed, desp)
            else:
                for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    eigv, time_elapsed, desp = utils.spectral_embeddings(matrix, i, measure_time=True, not_sparse=matrix_id == 'A')
                    time[matrix_id].append(time_elapsed)
                    print(time_elapsed, desp)
        print(time)
        utils.save_json(time, path=f'{args.dataset}/results/computational_difference_{args.dataset}_{args.version}.json')
    else:
        computational = utils.load_json(f'{args.dataset}/results/computational_difference_{args.dataset}_{args.version}.json')

        A = [computational['A'] for _ in range(9)]
        L = computational['L'][:9]
        N = computational['N'][:9]

        plt.plot(range(1, 10), A, label='A')
        plt.plot(range(1, 10), L, label='L')
        plt.plot(range(1, 10), N, label='N')

        plt.ylabel('Seconds', fontsize=16)
        plt.xlabel('#Eigenvectors', fontsize=16)
        plt.legend()
        plt.savefig(f'{args.dataset}/plots/{args.dataset}_{args.version}_computational_difference.jpg')

        plt.show()
