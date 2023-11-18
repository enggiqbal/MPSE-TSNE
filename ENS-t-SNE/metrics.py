import numpy as np  
import math 
from scipy.optimize import minimize_scalar

# def pairwise_distances(X: np.array):
#     x2 = np.sum(np.square(X), axis=1)
#     diff = np.sqrt( abs(x2.reshape(-1,1) - 2*np.dot(X,X.T) + x2 ))
#     np.fill_diagonal(diff, 0)
#     return diff

from sklearn.metrics import pairwise_distances

def stress(X,d):
    from math import comb
    N = len(X)
    ss = (X * X).sum(axis=1)

    diff = np.sqrt( abs(ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)) )

    np.fill_diagonal(diff,0)
    stress = lambda a:  np.sum( np.square( np.divide( (a*diff-d), d , out=np.zeros_like(d), where=d!=0) ) ) / comb(N,2)

    from scipy.optimize import minimize_scalar
    min_a = minimize_scalar(stress)
    # print("a is ",min_a.x)
    return stress(a=min_a.x)

def neighborhood_hit(X: np.array, d: np.array, k=7):
    diff = pairwise_distances(X)

    from sklearn.neighbors import NearestNeighbors as KNN
    ideal = KNN(n_neighbors=k,metric="precomputed").fit(d)
    real = KNN(n_neighbors=k,metric="precomputed").fit(diff)

    I = [set(x) for x in ideal.kneighbors()[1]]
    R = [set(x) for x in real.kneighbors()[1]]
    tot = sum( len(high.intersection(low)) for high,low in zip(I,R) )
    return tot / (X.shape[0] * k)
    
def kmean_acc(X: np.array, y: np.array, k=5):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=k).fit(X,y).score(X,y)

def silhouette_score(X:np.array, y: np.array):
    from sklearn.metrics import silhouette_score
    return silhouette_score(X,y)

def find_cluster_centers(X:np.array, y: np.array):
    unq, inv = np.unique(y,return_inverse=True)
    c_ids = [list() for _ in unq]
    [c_ids[inv[i]].append(i) for i,_ in enumerate(y)]
    cluster_centers = np.zeros( (len(c_ids), X.shape[1]) )
    for i,clusters in enumerate(c_ids):
        Xi = X[clusters]
        center = Xi.mean(axis=0)
        cluster_centers[i] = center 
    return cluster_centers

def cluster_distance(H:np.array, X:np.array, y:np.array):
    high_d_clusters = find_cluster_centers(H,y)
    low_d_clusters = find_cluster_centers(X,y)

    dh = pairwise_distances(high_d_clusters)
    return stress(low_d_clusters,dh)    

def within_class_distance_3d(data, labels):
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    within_class_distances = []
    total_sum_distances = 0
    
    for label in unique_labels:
        class_data = data[labels == label]
        class_centroid = np.mean(class_data, axis=0)
        class_distances = pairwise_distances(class_data, [class_centroid])
        within_class_distances.extend(class_distances)
        total_sum_distances += np.sum(class_distances)

    within_class_distance = np.mean(within_class_distances)
    total_sum_distance = total_sum_distances / len(labels)

    print(within_class_distance, total_sum_distance)

    return within_class_distance / np.max(pairwise_distances(data))

def squared_distance(point1, point2): 
    return np.sqrt( np.sum( np.square( point2-point1 ) ) )

def total_sum_of_squared_distances(data_points): 
    total_squared_distance = 0 
    for i in range(len(data_points)): 
        for j in range(i + 1, len(data_points)): 
            total_squared_distance += squared_distance(data_points[i], data_points[j]) 
    return total_squared_distance

# def within_class_distance_3d(data, labels): 
#     unique_labels = np.unique(labels) 
#     num_classes = len(unique_labels) 
#     within_class_distances = [] 
#     for label in unique_labels: 
#         class_data = data[labels == label] 
#         class_centroid = np.mean(class_data, axis=0) 
#         class_distances = pairwise_distances(class_data, [class_centroid]) 
#         within_class_distances.extend(class_distances) 
#     within_class_distance = np.mean(within_class_distances) 
#     return within_class_distance

def get_trustworthiness(high_d, low_d, k=5):
    from sklearn.manifold import trustworthiness
    return trustworthiness(high_d,low_d, n_neighbors=k)

def compute_all_metrics(X:np.array,d:np.array,y:np.array,full:np.array):
    # d = pairwise_distances(H)
    from pyDRMetrics.pyDRMetrics import DRMetrics
    import json
    drm = DRMetrics(full,X)
    drm_data = json.loads(drm.get_json())

    custom =  {
        "stress": stress(X,d),
        "NE":     1-neighborhood_hit(X,d),
        # "kmeans": 1-kmean_acc(X,y),
        "inverse silhouette":    1-silhouette_score(X,y),
        "trustworthiness" : get_trustworthiness(full,X)
        # "CD":     cluster_distance(H,X,y)
    }
    return drm_data | custom

def compute_all_3dmetrics(X: np.ndarray, d:np.ndarray, y:list[np.ndarray],full:np.ndarray):
    from pyDRMetrics.pyDRMetrics import DRMetrics
    import json
    drm = DRMetrics(full,X)
    drm_data = json.loads(drm.get_json())
    
    custom = {
        "stress": stress(X,d),
        "NE":     1-neighborhood_hit(X,d),
        # "kmeans": 1-kmean_acc(X,y),
        "CEV":    sum(within_class_distance_3d(X,yi)  for yi in y) / len(y),
        "trustworthiness": get_trustworthiness(full,X)
        # "CD":     cluster_distance(H,X,y)        
    }
    return drm_data | custom

def updated_metrics_2d(embedding, high_d, labels):
    from andreas_metrics import metric_continuity, metric_trustworthiness, metric_neighborhood_hit, metric_pq_normalized_stress
    D_low = pairwise_distances(embedding)

    results = {
        "continuity": metric_continuity(high_d, D_low, 7), 
        "trustworthiness": metric_trustworthiness(high_d, D_low, 7),
        "NH": metric_neighborhood_hit(embedding, labels, 7),
        "stress": stress(embedding,high_d)
    }
    return results


def updated_metrics_3d(embedding, high_d, labels):
    from andreas_metrics import metric_continuity, metric_trustworthiness, metric_neighborhood_hit, metric_pq_normalized_stress

    y = [tuple([labels[i][j] for i in range(len(labels))]) for j in range(labels[0].shape[0])]
    tuplemap = dict(zip(set(y), range(len(y))))
    y = np.array([tuplemap[x] for x in y])
    
    D_low = pairwise_distances(embedding)
    D_high = pairwise_distances(high_d)

    results = {
        "continuity": metric_continuity(D_high, D_low, 7), 
        "trustworthiness": metric_trustworthiness(D_high, D_low, 7),
        "NH": metric_neighborhood_hit(embedding, y, 7),
        "stress": stress(embedding,D_high)
    }
    return results



if __name__ == "__main__":
    from new_enstsne import load_penguins
    dists, labels, X = load_penguins()
    updated_metrics_3d(None, None, labels)

