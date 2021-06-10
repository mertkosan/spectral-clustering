import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import datasets
import learning
import copy

import argparse

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', help='Datasets: cora, email, ssets')
parser.add_argument('--version', default='1', help='version for ssets (1,2,3,4), default 1 for others')
parser.add_argument('--result', action='store_true', help='already have the results, want to plot')
args = parser.parse_args()


def shift_clusters(clusters, found_classes):
    unique_classes = set(np.unique(clusters))
    non_found_classes = list(unique_classes - found_classes)

    non_found_classes_original = copy.deepcopy(non_found_classes)
    np.random.shuffle(non_found_classes)
    m = {non_found_classes_original[i]: non_found_classes[i] for i in range(len(non_found_classes))}

    new_clusters = []
    for cluster in clusters:
        if cluster in m:
            new_clusters.append(m[cluster])
        else:
            new_clusters.append(cluster)

    return np.array(new_clusters)


def find_best_clusters(clusters, labels):
    trail_number = 500
    unique_classes = set(np.unique(labels))
    found_classes = set()

    best_clusters = clusters
    while len(found_classes) < len(unique_classes):
        clusters = best_clusters
        best_acc = accuracy_score(labels, clusters.astype(int))
        for trial in range(trail_number):
            shifted_clusters = shift_clusters(clusters, found_classes)
            acc = accuracy_score(shifted_clusters, labels)
            if acc > best_acc:
                print(acc)
                best_acc = acc
                best_clusters = shifted_clusters

        # check matching
        match_acc = 0
        match_number = 0
        for class_ in unique_classes - found_classes:
            pos = labels == class_
            acc = accuracy_score(labels[pos], best_clusters[pos])
            if acc > match_acc:
                match_number = class_
                match_acc = acc

        if match_number == 0:
            break

        found_classes.add(match_number)
        print(found_classes)

    return best_clusters


if __name__ == '__main__':

    if not args.result:
        if args.dataset == 'ssets':
            points = datasets.read_points(f'ssets/s{args.version}.txt')
            labels = datasets.read_labels(f'ssets/s{args.version}-label.pa')

            plt.scatter(points[:, 0], points[:, 1], c=labels)
            # plt.title(f'{args.dataset} V{args.version}'.upper())
            plt.axis('off')
            plt.savefig(f'{args.dataset}/graph_plots/{args.dataset}_{args.version}_graph.png', bbox_inches='tight')
            plt.show(bbox_inches='tight')

        else:
            A = datasets.load_graph_preprocessed(args.dataset, args.version)
            labels = datasets.load_labels_preprocessed(args.dataset, args.version)

            # graph draw
            G = nx.from_numpy_matrix(A.todense())
            layout = nx.spring_layout(G)
            nx.draw_networkx(G, pos=layout, node_color=labels, node_size=20, with_labels=False)
            # plt.title(f'{args.dataset}'.upper())
            plt.axis('off')
            plt.savefig(f'{args.dataset}/graph_plots/{args.dataset}_{args.version}_graph.png', bbox_inches='tight')
            plt.show()
    else:
        # best results
        best = {
            'cora_1': 25,
            'email_1': 24,
            'ssets_1': 13,
            'ssets_4': 8
        }

        if args.dataset == 'ssets':
            points = datasets.read_points(f'ssets/s{args.version}.txt')
            labels = datasets.read_labels(f'ssets/s{args.version}-label.pa')

            eigv = datasets.load_embeddings(args.dataset, args.version, 'N')[:, :best[f'{args.dataset}_{args.version}']]

            score, clusters = learning.k_means(eigv, labels, k=len(np.unique(labels)))

            best_clusters = find_best_clusters(clusters, labels)
            print(accuracy_score(labels, best_clusters))

            plt.scatter(points[:, 0], points[:, 1], c=labels)
            # plt.title(f'{args.dataset} V{args.version}'.upper())
            plt.axis('off')
            plt.savefig(f'{args.dataset}/graph_plots/{args.dataset}_{args.version}_graph.png', bbox_inches='tight')
            plt.show(bbox_inches='tight')

            plt.scatter(points[:, 0], points[:, 1], c=best_clusters)
            # plt.title(f'{args.dataset} V{args.version}'.upper())
            plt.axis('off')
            plt.savefig(f'{args.dataset}/graph_plots/{args.dataset}_{args.version}_graph_result.png', bbox_inches='tight')
            plt.show(bbox_inches='tight')
        else:
            A = datasets.load_graph_preprocessed(args.dataset, args.version)

            eigv = datasets.load_embeddings(args.dataset, args.version, 'N')[:, :best[f'{args.dataset}_{args.version}']]
            labels = datasets.load_labels_preprocessed(args.dataset, args.version)

            score, clusters = learning.k_means(eigv, labels, k=len(np.unique(labels)))

            best_clusters = find_best_clusters(clusters, labels)
            print(accuracy_score(labels, best_clusters))

            # original graph draw
            G = nx.from_numpy_matrix(A.todense())
            layout = nx.spring_layout(G)
            nx.draw_networkx(G, pos=layout, node_color=labels, node_size=20, with_labels=False)
            # plt.title(f'{args.dataset}'.upper())
            plt.axis('off')
            plt.savefig(f'{args.dataset}/graph_plots/{args.dataset}_{args.version}_graph.png', bbox_inches='tight')
            plt.show()

            # result graph draw
            nx.draw_networkx(G, pos=layout, node_color=best_clusters, node_size=20, with_labels=False)
            # plt.title(f'{args.dataset}'.upper())
            plt.axis('off')
            plt.savefig(f'{args.dataset}/graph_plots/{args.dataset}_{args.version}_graph_result.png', bbox_inches='tight')
            plt.show()
