import sys, os
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance

import misc, setup, multigraph, gd, projections, mds, tsne, plots, mpse, samples
from mpse import MPSE
from tsne import TSNE
import evaluate

def mpse_tsne(data, perplexity=30,
              iters=200, lr=[20,0.1],
              estimate_cost=True, evaluate=False,
              verbose=0, show_plots=False, save_results = False,output=None, smart_init=True,**kwargs):
    "Runs MPSE optimized for tsne"
    if verbose>0:
        print('***mview.mpse_tsne()***')

    #load/prepare distances and other variables from data
    if isinstance(data,str):
        import samples
        kwargs0 = kwargs
        distances, kwargs = samples.mload(data, verbose=verbose, **kwargs0)
        for key, value in kwargs0.items():
            kwargs[key] = value
    else:
        distances = data

    #init = avg_pca(data) if smart_init else None

    #start MPSE object
    mv =  MPSE(distances, visualization_method='tsne',
               visualization_args={'perplexity':perplexity,
                                   'estimate_cost':estimate_cost},
               verbose=verbose,
               indent='  ', **kwargs)
    mv.lr = lr
    n_samples = mv.n_samples

    #search for global minima
    mv.gd(fixed_projections=True, max_iter=20,
          scheme='bb')
    #mv.gd(fixed_projections=True, max_iter=20, batch_size=min(25,n_samples//2),
    #      scheme='mm')
    for divisor in [20,10,5]:
        batch_size = max(5,min(500,n_samples//divisor))
        mv.gd(batch_size=batch_size, max_iter=20, scheme='mm')
    mv.gd(max_iter=iters, batch_size = 32, scheme='mm')
   
    # mv.gd(max_iter=iters, scheme='mm')

    #mv.gd(max_iter=50, scheme='bb')
    #mv.gd(max_iter=200, scheme='fixed')

    if evaluate is True:
        mv.evaluate()

    #save outputs:
    if save_results is True:
        mv.save()

    #plots:
    if show_plots is True:
        mv.plot_computations()
        mv.plot_embedding()
        mv.plot_images(**kwargs)
        plt.show()

    if output:
        mv.plot_images(**kwargs)
        plt.savefig(output)

    return mv

def avg_pca(data):
    from sklearn.metrics import pairwise_distances
    from sklearn.decomposition import PCA
    n_perspectives = len(data)
    avg = np.zeros( (data[0].shape[0], data[0].shape[0]) )
    for x in data:
        avg += pairwise_distances(x)
    avg /= n_perspectives

    return PCA(n_components=3).fit_transform(avg)

def compare_tsne(data,
                 estimate_cost=True, evaluate=False,
                 save_results=False, show_plots=True, **kwargs):
    "compares mpse-tsne to regular tsne"

    #load/prepare distances and other variables from data
    if isinstance(data,str):
        import samples
        distances, lkwargs = samples.load(data, dataset_type='both', **kwargs)
    else:
        distances = data
        lkwargs = [kwargs]*(len(data)+1)

    results = []
    n_perspectives = len(distances)
    for i in range(n_perspectives):
        kwargsi = lkwargs[i+1]
        for key, value in kwargs.items():
            kwargsi[key] = value
        ts = TSNE(distances[i], **kwargsi)
        ts.optimized()
        if evaluate is True:
            ts.evaluate()
        results.append(ts)
        if show_plots:
            ts.plot_embedding()
    kwargs0 = lkwargs[0]
    for key, value in kwargs.items():
        kwargs0[key] = value
    mv=mpse_tsne(distances,
                 estimate_cost=estimate_cost, evaluate=evaluate,
                 **kwargs0)
    results.insert(0,mv)
    if show_plots:
        mv.plot_embedding()
        mv.plot_images()
        plt.show()
    return results

def compare_perplexity(data, perplexities=[30,200],iters=100,
                       save_results=True, **kwargs):
    "runs mpse_tsne on the same perspective w/ different perplexity values"

    if save_results is True:
        "save results to csv files"
        directory = os.path.dirname(os.path.realpath(__file__))
        sys.path.insert(1, directory)
        location=directory+'/temp/'
        if not os.path.exists(location):
            os.makedirs(location)
        for f in os.listdir(location):
            os.remove(os.path.join(location,f))


    #load/prepare distances and other variables from data
    if isinstance(data,str):
        import samples
        kwargs0 = kwargs
        distances, kwargs = samples.sload(data, **kwargs0)
        for key, value in kwargs0.items():
            kwargs[key] = value
    else:
        distances = data

    #prepare figures
    K = len(perplexities)
    fig, ax = plt.subplots(2,K, figsize=(3*K,3))

    for i,p in enumerate(perplexities):
        ts = TSNE(distances,perplexity=p, **kwargs)
        ts.optimized()
        #ts.gd()
        ts.plot_embedding(axis=False,ax=ax[0,i])
        print('tsne cost for perspective',i,':',ts.cost)
        if save_results is True:
            np.savetxt(location+'tsne_image_'+str(i)+'.csv',
                       ts.embedding)

    distances = [distances]*len(perplexities)


    va = []
    for p in perplexities:
        va.append({'perplexity':p})
    mv = MPSE(distances,visualization_method='tsne',
              visualization_args=va,
              verbose=0,**kwargs)

    #search for global minima
    n_samples = mv.n_samples
    mv.gd(fixed_projections=True, max_iter=10,
          scheme='bb')
    for divisor in [20,10,5]:
        batch_size = max(5,min(500,n_samples//divisor))
        mv.gd(batch_size=batch_size, max_iter=20, scheme='mm')
    mv.gd(max_iter=iters, batch_size=int(n_samples/2), scheme='mm')
    #mv.gd(max_iter=20, scheme='fixed')
    print(mv.individual_cost)

    if save_results is True:
        np.savetxt(location+'embedding.csv', mv.embedding)
        for i in range(mv.n_perspectives):
            np.savetxt(location+'projection_'+str(i)+'.csv',
                       mv.projections[i])
            np.savetxt(location+'images_'+str(i)+'.csv',
                       mv.images[i])
        np.savetxt(location+'cost_history.csv', mv.cost_history)
        if mv.sample_classes is not None:
            np.savetxt(location+'sample_classes.csv', mv.sample_classes,
                       fmt='%d')
        if mv.image_classes is not None:
            for i in range(mv.n_perspectives):
                np.savetxt(location+'image_classes_'+str(i)+'.csv',
                           mv.image_classes[i])

    mv.plot_images(axis=False, ax=ax[1])
    plt.axis('equal')
    mv.plot_computations()
    mv.plot_embedding()

    plt.draw()
    plt.pause(0.2)
    plt.show()

    return

if __name__=='__main__':

    estimate_cost=False
    evaluate=True

#    mpse_tsne('narrow2',n_samples=300,perplexity=50,
#              n_perspectives=2, estimate_cost=False,
#              verbose=2,show_plots=True,save_results=False)

#    mpse_tsne('pride',n_samples=300,perplexity=30,
#              n_perspectives=3, estimate_cost=False,
#              verbose=2,show_plots=True,save_results=False)


    run_all_mpse_tsne = False
    if run_all_mpse_tsne is True:
        mpse_tsne('equidistant',
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('disk', n_perspectives=10,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('clusters', n_clusters=[3,4,2],
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('clusters2', n_clusters=2, n_perspectives=4, perplexity=80,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('florence', perplexity = 40,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('123', n_samples=300, perplexity = 250,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('mnist',n_samples=400,perplexity=30,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('mnist',n_samples=400,perplexity=100,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('phishing',
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)

    run_all_compare_tsne = False
    if run_all_compare_tsne is True:
        compare_tsne('clusters', n_samples=400, n_perspectives=1,
                     evaluate=True, verbose=2)
        compare_tsne('clusters', n_samples=400, n_perspectives=2,
                     evaluate=True, verbose=2)

    run_all_compare_perplexity = True
    if run_all_compare_perplexity is True:
        compare_perplexity('clusters', n_samples=400,
                           n_clusters=2, n_perspectives=5,
                           perplexities=[30,30,30,30,30])
