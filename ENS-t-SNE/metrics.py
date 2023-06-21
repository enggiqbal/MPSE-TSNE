import numpy as np  
import math 
from scipy.optimize import minimize_scalar

def pairwise_distances(X: np.array):
    x2 = np.sum(np.square(X), axis=1)
    diff = np.sqrt( abs(x2.reshape(-1,1) - 2*np.dot(X,X.T) + x2 ))
    np.fill_diagonal(diff, 0)
    return diff

def stress(X,d):
    from math import comb
    N = len(X)
    ss = (X * X).sum(axis=1)

    diff = np.sqrt( abs(ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)) )

    np.fill_diagonal(diff,0)
    stress = lambda a:  np.sum( np.square( np.divide( (a*diff-d), d , out=np.zeros_like(d), where=d!=0) ) ) / comb(N,2)

    from scipy.optimize import minimize_scalar
    min_a = minimize_scalar(stress)
    #print("a is ",min_a.x)
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

def compute_all_metrics(X:np.array,H:np.array,y:np.array):
    d = pairwise_distances(H)
    return {
        "stress": stress(X,d),
        "NH":     1-neighborhood_hit(X,d),
        "kmeans": 1-kmean_acc(X,y),
        "sil":    silhouette_score(X,y),
        "CD":     cluster_distance(H,X,y)
    }

if __name__ == "__main__":
    Y = np.random.uniform(-1,1,(100,5))

    from sklearn.manifold import TSNE
    X = TSNE().fit_transform(Y)

    s = neighborhood_hit(X,pairwise_distances(Y))

