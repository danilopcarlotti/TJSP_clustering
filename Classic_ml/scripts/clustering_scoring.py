from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.cluster import homogeneity_score
from sklearn.preprocessing import MinMaxScaler


import numpy as np


def cluster_items(
    X: np.array, items: list, y: list = [], n_clusters: int = 10, to_scale: bool = True
):
    """
    Returns a list of tuples (item, number of similar items)
    from a X matrix and a list of items
    """
    if to_scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels_dict = Counter(kmeans_model.labels_)
    if len(y):
        hom_score = homogeneity_score(y, kmeans_model.labels_)
    else:
        hom_score = -1
    closest, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, X)
    return [
        (items[i], labels_dict[kmeans_model.labels_[i]]) for i in set(closest)
    ], hom_score
