from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.cluster import homogeneity_score
from sklearn.preprocessing import MinMaxScaler


import numpy as np


def cluster_items(
    X: np.array, items: list, y: list = [], n_clusters: int = 20, to_scale: bool = True
):
    """
    Returns a list of tuples (item, percentage of similar items)
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
    total = len(X)
    m_clusters = kmeans_model.labels_.tolist()
    closest_data = []
    for i in range(n_clusters):
        center_vec = kmeans_model.cluster_centers_[i]
        data_idx_within_i_cluster = [
            idx for idx, clu_num in enumerate(m_clusters) if clu_num == i
        ]
        one_cluster_tf_matrix = np.zeros(
            (len(data_idx_within_i_cluster), kmeans_model.cluster_centers_.shape[1])
        )
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_cluster_tf_matrix[row_num] = X[data_idx]
        closest, _ = pairwise_distances_argmin_min([center_vec], one_cluster_tf_matrix)
        closest_idx_in_one_cluster_tf_matrix = closest[0]
        closest_data_row_num = data_idx_within_i_cluster[
            closest_idx_in_one_cluster_tf_matrix
        ]
        closest_data.append(closest_data_row_num)
    closest_data = list(set(closest_data))
    return [
        (
            items[i],
            "{:.2f}%".format(100 * (labels_dict[kmeans_model.labels_[i]] / total)),
        )
        for i in set(closest_data)
    ], hom_score
