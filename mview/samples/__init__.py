import os, sys
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1,directory)
import csv

import numpy as np

def mnist(n_samples=1000, digits=None, **kwargs):
    from keras.datasets import mnist
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    if digits is not None:
        indices = [i for i in range(len(Y_train)) if Y_train[i] in digits]
        X_train = X_train[indices]
        Y_train = Y_train[indices]
    if isinstance(n_samples,int):
        indices = list(range(0,n_samples))
    else:
        indices = n_samples
    assert(len(X_train)) >= len(indices)
    X = X_train[indices]
    labels = Y_train[indices]

    X = X.reshape(len(indices),28*28)

    #save original:
    "save results to csv files"
    location=directory+'/../temp2/'
    if not os.path.exists(location):
        os.makedirs(location)
    for f in os.listdir(location):
        os.remove(os.path.join(location,f))
    np.savetxt(location+'mnist_original_images.csv',X)
    np.savetxt(location+'mnist_original_labels.csv',labels)
    
    X = np.array(X,dtype='float')/256
    return X, labels

def sload(dataset, n_samples=200, **kwargs):
    "returns a distance array and dictionary with aditional features"
    data = {}
    if dataset == 'equidistant':
        length = n_samples*(n_samples-1)//2
        distances = np.random.normal(1,0.1,length)
        data['sample_colors'] = n_samples-1
    elif dataset == 'disk':
        import misc
        distances = misc.disk(n_samples, dim=2)
        data['features'] = distances
        data['sample_colors'] = n_samples-1
    elif dataset == 'disk2':
        import misc
        distances = misc.disk(n_samples, dim=2)
        data['features'] = distances
        data['sample_colors'] = [0 if distances[i,0]<0 else 1 for i in range(n_samples)]
    elif dataset == 'clusters':
        from clusters import clusters
        if 'n_clusters' in kwargs:
            n_clusters = kwargs.pop('n_clusters')
        else:
            n_clusters = 2
        distances, sc = clusters(n_samples, n_clusters=n_clusters, **kwargs)
        data['sample_classes'] = sc
        data['sample_colors'] = data['sample_classes']
    elif dataset == 'clusters2':
        from clusters import clusters2
        if 'n_clusters' in kwargs:
            n_clusters = kwargs.pop('n_clusters')
        else:
            n_clusters = 2
        distances, sc = clusters2(n_samples, n_clusters=n_clusters, **kwargs)
        data['sample_classes'] = sc
        data['sample_colors'] = data['sample_classes']
    elif dataset == 'narrow':
        from clusters import narrow
        d, data['sample_classes'] = narrow(n_samples, n_perspectives=2)
        distances = d[0]
        data['sample_colors'] = data['sample_classes']
    elif dataset == 'mnist':
        data['features'], data['sample_classes'] = mnist(**kwargs)
        distances = data['features']
        data['sample_colors'] = data['sample_classes']
    elif dataset in ['pride','pride_and_prejudice']:
        import pride_and_prejudice
        distances = pride_and_prejudice.distances[2]
        data['edges'] = pride_and_prejudice.edges[2]
    else:
        print('***dataset not found***')

    return distances, data

def mload(dataset, n_samples=100, n_perspectives=2, **kwargs):
    "returns dictionary with datasets"

    distances = []
    data = {}
    if dataset == 'equidistant':
        length = n_samples*(n_samples-1)//2
        for persp in range(n_perspectives):      
            distances.append(np.random.normal(1,0.1,length))
        data['image_colors'] = [n_samples-1]*n_perspectives
    elif dataset == 'disk':
        import misc, projections
        X = misc.disk(n_samples, dim=3)
        proj = projections.PROJ()
        Q = proj.generate(number=n_perspectives, method='random')
        Y = proj.project(Q,X)
        data['true_images'] = Y
        data['true_embedding'] = X
        data['true_projections'] = Q
        distances = Y
        data['image_colors'] = [0]*n_perspectives
    elif dataset == 'disk2':
        import misc, projections
        X = misc.disk(n_samples, dim=3)
        proj = projections.PROJ()
        Q = proj.generate(number=n_perspectives, method='random')
        Y = proj.project(Q,X)
        data['true_images'] = Y
        data['true_embedding'] = X
        data['true_projections'] = Q
        distances = Y
        labels = []
        for y in Y:
            labels.append([1 if yi[0]>=0 else 0 for yi in y])
        data['image_classes'] = labels
        data['image_colors'] = data['image_classes']
    elif dataset == 'clusters2a':
        from clusters import createClusters
        D, data['image_colors'] = \
            createClusters(n_samples, n_perspectives)
    elif dataset == 'clusters':
        from clusters import clusters
        distances = []
        data['image_classes'] = []
        data['image_colors'] = []
        if 'n_clusters' in kwargs:
            n_clusters = kwargs.pop('n_clusters')
        else:
            n_clusters = 2
        if isinstance(n_clusters,int):
            n_clusters = [n_clusters]*n_perspectives
        else:
            n_perspectives = len(n_clusters)
        for i in range(n_perspectives):
            d, c = clusters(n_samples, n_clusters=n_clusters[i], **kwargs)
            distances.append(d)
            data['image_classes'].append(c)
            data['image_colors'].append(c)
    elif dataset == 'clusters2':
        from clusters import clusters2
        distances = []; data['image_colors'] = []
        if 'n_clusters' in kwargs:
            n_clusters = kwargs['n_clusters']
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]*n_perspectives
        for persp in range(n_perspectives):
            d, c = clusters2(n_samples,n_clusters[persp])
            distances.append(d); data['image_colors'].append(c)
        data['image_classes'] = data['image_colors']
    elif dataset == 'narrow':
        from clusters import narrow
        distances, data['sample_classes'] = narrow(n_samples, n_perspectives)
        data['sample_colors'] = data['sample_classes']
    elif dataset == '123':
        import projections
        X = np.genfromtxt(directory+'/123/123.csv',delimiter=',')[0:n_samples]
        X1 = np.genfromtxt(directory+'/123/1.csv',delimiter=',')[0:n_samples]
        X2 = np.genfromtxt(directory+'/123/2.csv',delimiter=',')[0:n_samples]
        X3 = np.genfromtxt(directory+'/123/3.csv',delimiter=',')[0:n_samples]
        proj = projections.PROJ()
        Q = proj.generate(number=3,method='cylinder')
        distances = [X1,X2,X3]
        data['true_embedding'] = X
        data['true_projections'] = Q
        data['true_images'] = [X1,X2,X3]
        data['image_colors'] = [0]*3
    elif dataset == 'florence':
        import florence
        distances, dictf = florence.setup()
        for key, value in dictf.items():
            data[key] = value
    elif dataset == 'credit':
        import csv
        path = directory+'/credit/'
        Y = []
        for ind in ['1','2','3']:
            filec = open(path+'discredit3_tsne_cluster_1000_'+ind+'.csv')
            array = np.array(list(csv.reader(filec)),dtype='float')
            array += np.random.randn(len(array),len(array))*1e-4
            Y.append(array)
        distances = Y
    elif dataset == 'phishing':
        import phishing
        features = phishing.features
        labels = phishing.group_names
        if n_samples is None:
            n_samples = len(features[0])
        Y, perspective_labels = [], []
        for group in [0,1,2,3]:
            assert group in [0,1,2,3]
            Y.append(features[group][0:n_samples])
            perspective_labels.append(labels[group])
        sample_colors = phishing.results[0:n_samples]
        distances = Y
        data['sample_colors'] = sample_colors
        data['perspective_labels'] = perspective_labels
        data['sample_classes'] = sample_colors
    elif dataset == 'mnist':
        X, data['sample_classes'] = mnist(n_samples, **kwargs)
        data['features'] = X
        distances = [X[:,0:28*14],X[:,28*14::]]
        data['sample_colors'] = data['sample_classes'].copy()
    elif dataset in ['pride','pride_and_prejudice']:
        import pride_and_prejudice
        distances = pride_and_prejudice.distances
        data['edges'] = pride_and_prejudice.edges
        data['sample_labels'] = pride_and_prejudice.names
    else:
        print('***dataset not found***')
    return distances, data

def load(dataset, dataset_type='multiple',**kwargs):
    "loads dataset for single or multiple or both perspectives"
    if dataset_type=='single':
        return sload(dataset, **kwargs)
    elif dataset_type=='multiple':
        return mload(dataset, **kwargs)
    elif dataset_type=='both':
        distances, kwargs0 = mload(dataset, **kwargs)
        n_perspectives = len(distances)
        kwargs = [kwargs0]
        for i in range(n_perspectives):
            kwargsi = kwargs0.copy()
            if 'image_colors' in kwargsi:
                kwargsi['sample_colors']=kwargsi['image_colors'][i]
            if 'image_labels' in kwargsi:
                kwargsi['sample_labels']=kwargsi['image_labels'][i]
            if 'image_colors' in kwargsi:
                kwargsi['sample_classes']=kwargsi['image_classes'][i]
            kwargs.append(kwargsi)
        return distances, kwargs
    else:
        return
