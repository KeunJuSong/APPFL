import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from itertools import product, permutations

def get_num_cluster(gradient_profile_matrix, n_centers, n_clients, estimated_cluster_ids_old=None):
    start_time = time.time()

    P, singular_values, Q = np.linalg.svd(gradient_profile_matrix, full_matrices=False)
    print("singular_values:", singular_values)
    singular_gaps = singular_values[:-1] - singular_values[1:]
    print("singular_gaps:", singular_gaps)
    num_of_leading_sv = 1 + np.argmax(singular_gaps)
    clipped_num_of_leading_sv = np.clip(num_of_leading_sv, a_min=3, a_max=n_clients)
    print("num of leading singular values (min clipping 3) / clipped: ", num_of_leading_sv, " / ",
          clipped_num_of_leading_sv)

    reduced_G = np.matmul(np.transpose(P[:, :clipped_num_of_leading_sv]), gradient_profile_matrix)
    silhouette_avg_list = []
    for k in np.arange(2, n_centers + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)  # .fit(np.transpose(reduced_G))
        cluster_labels = kmeans.fit_predict(np.transpose(reduced_G))
        silhouette_avg = silhouette_score(np.transpose(reduced_G), cluster_labels)
        silhouette_avg_list.append(silhouette_avg)

    print("silhouette_avg_list:", silhouette_avg_list)
    proposed_k = np.argmax(silhouette_avg_list) + 2
    return proposed_k

def spectral_clustering_and_matching(gradient_profile_matrix, n_centers, n_clients, estimated_cluster_ids_old=None,
                                     clustering_algorithm='KMeans'):
    start_time = time.time()
    P, singular_values, Q = np.linalg.svd(gradient_profile_matrix, full_matrices=False)
    reduced_G = np.matmul(np.transpose(P[:, :n_centers]), gradient_profile_matrix)
    SVD_time = time.time() - start_time
    # print("SVD takes ", time.time()-start_time, " sec.")

    start_time = time.time()
    estimated_part_ids = None
    if clustering_algorithm == "KMeans":
        kmeans = KMeans(n_clusters=n_centers, init="k-means++", random_state=42).fit(np.transpose(reduced_G))
        cluster_centers = kmeans.cluster_centers_
        estimated_part_ids = kmeans.labels_
    elif clustering_algorithm == "DBSCAN":
        dbscan = DBSCAN(eps=0.7, min_samples=2, leaf_size=10).fit(np.transpose(reduced_G))
        estimated_part_ids = dbscan.labels_
    elif clustering_algorithm == "AgglomerativeClustering":
        ac = AgglomerativeClustering(n_clusters=n_centers).fit(np.transpose(reduced_G))
        estimated_part_ids = ac.labels_

    K_means_clustering_time = time.time() - start_time

    start_time = time.time()
    if estimated_cluster_ids_old is not None:
        best_estimated_part_ids = estimated_part_ids
        best_n_consistency = 0
        for model_order in list(permutations([m_idx for m_idx in range(n_centers)])):
            temp_estimated_part_ids = np.array(model_order)[estimated_part_ids]
            n_consistency = np.sum((estimated_cluster_ids_old - temp_estimated_part_ids) == 0)
            if best_n_consistency < n_consistency:
                best_estimated_part_ids = temp_estimated_part_ids
                best_n_consistency = n_consistency
    else:
        best_estimated_part_ids = estimated_part_ids
    index_matching_time = time.time() - start_time

    info = {
        "estimated_cluster_ids": best_estimated_part_ids,
        "reduced_gradient_profile_matrix": reduced_G,
        "singular_values": singular_values,
        "SVD_time": SVD_time,
        "K_means_clustering_time": K_means_clustering_time,
        "index_matching_time": index_matching_time
    }
    return info