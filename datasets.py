import numpy as np
import scipy.sparse as sp

import networkx as nx

from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler


# Cora
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_cora():
    # cora dataset and loading function from https://github.com/tkipf/pygcn
    idx_features_labels = np.genfromtxt("cora/cora.content", dtype=np.dtype(str))
    labels = encode_onehot(idx_features_labels[:, -1])
    labels = np.array([np.argmax(label) + 1 for label in labels])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("cora/cora.cites", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    return adj, labels


def load_graph_preprocessed(dataset, version):
    return np.load(f'{dataset}/{dataset}_v{version}_graph_preprocessed.npy', allow_pickle=True).item()


def load_labels_preprocessed(dataset, version):
    return np.load(f'{dataset}/{dataset}_v{version}_labels_preprocessed.npy')


def load_embeddings(dataset, version, matrix_id):
    return np.load(f'{dataset}/embeddings/{matrix_id}_{dataset}_v{version}.npy')


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# Stanford Email dataset: http://snap.stanford.edu/data/email-Eu-core.html
def read_stanford_graph(path):
    adj = nx.to_numpy_array(nx.read_edgelist(path, nodetype=int))
    adj = sp.coo_matrix(adj)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj


def read_stanford_label(path, graph_size):
    labels = [0] * graph_size
    with open(path) as _file:
        for line in _file.readlines():
            labels[int(line.split()[0])] = int(line.split()[1])
    return np.array(labels)


# Ssets: http://cs.joensuu.fi/sipu/datasets/
def read_points(path):
    points = []
    with open(path) as _file:
        for line in _file.readlines():
            p = line.strip().split()
            points.append(list(map(int, p)))
    return np.array(points)


def read_labels(path):
    labels = []
    with open(path) as _file:
        for line in _file.readlines():
            labels.append(int(line.strip()))
    return np.array(labels)


def load_graph(dataset, version):
    if dataset == 'cora':
        adj, labels = load_cora()

        # select largest connected
        G = nx.from_numpy_array(np.array(adj.todense()))
        c = list(nx.connected_components(G))[0]
        adj = adj[list(c), :][:, list(c)]
        labels = labels[list(c)]

        np.save(f'{dataset}/{dataset}_v{version}_graph_preprocessed.npy', adj)
        np.save(f'{dataset}/{dataset}_v{version}_labels_preprocessed.npy', labels)

        return adj, labels

    elif dataset == 'email':
        adj = read_stanford_graph('email/email-Eu-core.txt')
        labels = read_stanford_label('email/email-Eu-core-department-labels.txt', adj.shape[0])

        # select largest connected
        G = nx.from_numpy_array(np.array(adj.todense()))
        c = list(nx.connected_components(G))[0]
        adj = adj[list(c), :][:, list(c)]
        labels = labels[list(c)]

        np.save(f'{dataset}/{dataset}_v{version}_graph_preprocessed.npy', adj)
        np.save(f'{dataset}/{dataset}_v{version}_labels_preprocessed.npy', labels)

        return sp.coo_matrix(adj), labels
    elif dataset == 'ssets':
        points = read_points(f'ssets/s{version}.txt')
        labels = read_labels(f'ssets/s{version}-label.pa')

        points = StandardScaler().fit_transform(points)

        rbf = RBF()
        adj = rbf(points)

        np.save(f'{dataset}/{dataset}_v{version}_graph_preprocessed.npy', adj)
        np.save(f'{dataset}/{dataset}_v{version}_labels_preprocessed.npy', labels)

        return adj, labels
