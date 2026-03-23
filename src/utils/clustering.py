import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import kmedoids

#--------------------- K Means -----------------
# Calculate optimal number of clusters based on distance threshold
def calc_clusters(Z_embedding, k_min=1, k_max=10, dist_threshold=1, sample_weight=None, seed=42):
    # Update the k_max if it's equal to k_min
    if k_min == k_max:
        k_max = 2*k_max
    k = k_max
    while k > k_min:
        k_means = KMeans(n_clusters=k, random_state=seed).fit(Z_embedding, sample_weight=sample_weight)
        centroid_dist = cdist(k_means.cluster_centers_, k_means.cluster_centers_, metric='euclidean')
        centroid_dist += np.eye(k)*dist_threshold  # ignore zero distances on diagonal
        if np.any(centroid_dist < dist_threshold):
            k -= 1
            continue
        break
    return k


# K-means clustering function
def k_means_clustering(Z_embedding, k_min=1, k_max=10, dist_threshold=1, sample_weight=None, seed=42):
    k = calc_clusters(
        Z_embedding,
        k_min=k_min, 
        k_max=k_max, 
        dist_threshold=dist_threshold, 
        sample_weight=sample_weight, 
        seed=seed
    )
    kmeans_model = KMeans(n_clusters=k, random_state=seed).fit(Z_embedding,
                                                               sample_weight=sample_weight)
    return kmeans_model, kmeans_model.cluster_centers_


#--------------------- K Means -----------------
# Calculate optimal number of clusters based on distance threshold
def calc_clusters_kmedoids(Z_embedding, k_min=1, k_max=10, dist_threshold=1, seed=42, distance_metric='euclidean'):
    # Update the k_max if it's equal to k_min
    if k_min == k_max:
        k_max = 2*k_max
    k = k_max
    if distance_metric == 'euclidean':
        diss = euclidean_distances(Z_embedding)
    elif distance_metric == 'cosine':
        diss = 1 - cosine_similarity(Z_embedding)
    else:
        raise ValueError(f"Unknown distance metric '{distance_metric}'")
    while k >= k_min:
        kmedoids_model = kmedoids.KMedoids(n_clusters=k, random_state=seed)       
        km = kmedoids_model.fit(diss)
        medoids = Z_embedding[km.medoid_indices_]
        if distance_metric == 'euclidean':
            medoid_dist = euclidean_distances(medoids)
            medoid_dist += np.eye(k)*dist_threshold  # ignore zero distances on diagonal
        elif distance_metric == 'cosine':
            medoid_dist = 1 - cosine_similarity(medoids)
            medoid_dist += np.eye(k)*dist_threshold  # ignore zero distances on diagonal
        if np.any(medoid_dist < dist_threshold):
            k -= 1
            continue
        break
    return kmedoids_model, medoids

