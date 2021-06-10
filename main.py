import networkx as nx
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt

import datasets
import utils
import learning

import argparse
import time

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', help='Datasets: cora, email, ssets')
parser.add_argument('--version', default='1', help='version for ssets, default 1 for others')
parser.add_argument('--task', default='clustering', help='options: clustering, classification')
parser.add_argument('--method', default='nn', help='svm, nn, forest')
parser.add_argument('--smallest', action='store_true')
parser.add_argument('--plot', action='store_true', help='already have the results, want to plot.')
args = parser.parse_args()

if __name__ == '__main__':
    smallest = args.smallest
    # TODO: change here based on experiments
    rang = [2, 4, 6, 16, 32, 64, 128, 256]

    if not args.plot:
        embeds_A = datasets.load_embeddings(args.dataset, args.version, 'A')
        embeds_L = datasets.load_embeddings(args.dataset, args.version, 'L')
        embeds_N = datasets.load_embeddings(args.dataset, args.version, 'N')

        labels = datasets.load_labels_preprocessed(args.dataset, args.version)

        if args.task == 'classification' and args.method == 'nn':
            labels = datasets.encode_onehot(labels)

        embeds = {
            'A': embeds_A,
            'L': embeds_L,
            'N': embeds_N
        }

        results = {
            'A': [],
            'L': [],
            'N': []
        }
        for matrix_id in embeds:
            embeds_ = embeds[matrix_id]
            start_time = time.time()
            for i in rang:
                eigv = embeds_[:, :i] if smallest else embeds_[:, -i:]
                trials = []
                for trial in range(5):
                    if args.task == 'clustering':
                        score, clusters = learning.k_means(eigv, labels, k=len(np.unique(labels)))
                    elif args.task == 'classification':
                        if args.method == 'svm':
                            train_score, score = learning.svm(eigv, labels)
                        elif args.method == 'nn':
                            train_score, score = learning.nn(eigv, labels)
                        elif args.method == 'forest':
                            train_score, score = learning.forest(eigv, labels)
                    else:
                        raise Exception(f'No task: {args.task}')
                    trials.append(score)
                score_avg = sum(trials) / len(trials)
                results[matrix_id].append(score_avg)
                print(score_avg)
            end_time = time.time()
            results[f'{matrix_id}_time'] = end_time - start_time

        utils.save_json(results, f'{args.dataset}/results/{args.dataset}_v{args.version}_{args.task}{"_" + args.method if args.task == "classification" else ""}'
                                 f'_results_{"S" if smallest else "L"}.json')
    else:
        results = utils.load_json(f'{args.dataset}/results/{args.dataset}_v{args.version}_{args.task}{"_" + args.method if args.task == "classification" else ""}'
                                  f'_results_{"S" if smallest else "L"}.json')

        plt.semilogx(rang, results['A'], label='A')
        plt.semilogx(rang, results['L'], label='L')
        plt.semilogx(rang, results['N'], label='N')

        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('#Eigenvectors', fontsize=16)

        plt.legend()
        plt.savefig(f'{args.dataset}/plots/{args.dataset}_{args.version}_{args.task}{"_" + args.method if args.task == "classification" else ""}'
                    f'_{"S" if smallest else "L"}.jpg')
        plt.show()

        # results_nn = utils.load_json(f'{args.dataset}/{args.dataset}_v{args.version}_{args.task}_nn'
        #                              f'_results_{"S" if smallest else "L"}.json')
        # results_svm = utils.load_json(f'{args.dataset}/{args.dataset}_v{args.version}_{args.task}_svm'
        #                               f'_results_{"S" if smallest else "L"}.json')
        # results_forest = utils.load_json(f'{args.dataset}/{args.dataset}_v{args.version}_{args.task}_forest'
        #                                  f'_results_{"S" if smallest else "L"}.json')
        #
        # plt.semilogx(rang, results_nn['N'], label='Neural Networks')
        # plt.semilogx(rang, results_svm['N'], label='SVM')
        # plt.semilogx(rang, results_forest['N'], label='Random Forest')
        #
        # plt.ylabel('Accuracy', fontsize=16)
        # plt.xlabel('#Eigenvectors', fontsize=16)
        #
        # plt.legend()
        # plt.savefig(f'{args.dataset}/{args.dataset}_{args.version}_{args.task}_comparison'
        #             f'_{"S" if smallest else "L"}.jpg')
        # plt.show()
