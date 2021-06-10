import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import random
from sklearn.metrics import normalized_mutual_info_score


def nn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=3000))
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    score = clf.score(X_test, y_test)

    return train_score, score


def svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    score = clf.score(X_test, y_test)

    return train_score, score


def forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=2))
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    score = clf.score(X_test, y_test)

    return train_score, score


def k_means(points: np.array, labels, k: int) -> np.array:
    restart = True
    while restart:
        try:
            restart = False
            n, m = points.shape

            change_happened = True
            centroids = np.array([points[i] for i in random.sample(range(n), k)])
            clusters = np.array([None for _ in range(n)])
            while change_happened:
                change_happened = False

                # assign clusters
                for i, point in enumerate(points):
                    c = np.argmin(np.linalg.norm(centroids - point, axis=-1)) + 1
                    if clusters[i] != c:
                        clusters[i] = c
                        change_happened = True

                for i in range(1, k + 1):
                    clusters_i = points[clusters == i]
                    if len(clusters_i) == 0:
                        restart = True
                        raise Exception('mean slice is empty.')
                    centroids[i - 1] = np.mean(clusters_i, axis=0)
        except Exception as e:
            pass

    nmi = normalized_mutual_info_score(labels, clusters)

    return nmi, clusters
