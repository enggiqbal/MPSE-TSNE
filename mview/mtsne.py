import sys
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
              iters=100, lr=[20,0.1],
              estimate_cost=True, evaluate=False,
              verbose=0, show_plots=False, save_results = False,**kwargs):
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
        
    #start MPSE object
    mv =  MPSE(distances, visualization_method='tsne',
               visualization_args={'perplexity':perplexity,
                                   'estimate_cost':estimate_cost},
               verbose=verbose,
               indent='  ', **kwargs)
    mv.lr = lr
    n_samples = mv.n_samples

    #search for global minima
    mv.gd(fixed_projections=True, max_iter=10, batch_size=min(25,n_samples//2),
          scheme='mm')
    for divisor in [20,10,5]:
        batch_size = max(5,min(500,n_samples//divisor))
        mv.gd(batch_size=batch_size, max_iter=20, scheme='mm')
    mv.gd(max_iter=iters, batch_size=int(n_samples/2), scheme='mm')
    #mv.gd(max_iter=20, scheme='bb')
    #mv.gd(max_iter=iters, scheme='fixed')

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
        
    return mv

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

def compare_perplexity(data, perplexities=[30,200],iters=50, **kwargs):
    "runs mpse_tsne on the same perspective w/ different perplexity values"
    
    #load/prepare distances and other variables from data
    if isinstance(data,str):
        import samples
        kwargs0 = kwargs
        distances, kwargs = samples.sload(data, **kwargs0)
        for key, value in kwargs0.items():
            kwargs[key] = value
    else:
        distances = data

    for p in perplexities:
        ts = TSNE(distances,perplexity=p, **kwargs)
        ts.optimized()
        ts.plot_embedding()
        print(ts.cost)
        
    distances = [distances]*len(perplexities)

    va = []
    for p in perplexities:
        va.append({'perplexity':p})
    mv = MPSE(distances,visualization_method='tsne',
              visualization_args=va,
              verbose=0,**kwargs)

    #search for global minima
    n_samples=200
    mv.gd(fixed_projections=True, max_iter=50, batch_size=min(25,n_samples//2),
          scheme='mm')
    for divisor in [20,10,5,2]:
        batch_size = max(5,min(500,n_samples//divisor))
        mv.gd(batch_size=batch_size, max_iter=20, scheme='mm')
    #mv.gd(max_iter=20, scheme='mm')
    mv.gd(max_iter=iters, scheme='fixed')
    print(mv.individual_cost)
        
    mv.plot_computations()
    mv.plot_embedding(title='final embeding')
    mv.plot_images()
    plt.draw()
    plt.pause(0.2)
    plt.show()
    
    return

def compare_mds_tsne(dataset='mnist', perplexity=30):
    data = samples.load(dataset)
    D = [data['X']]*2
    va = {'perplexity':perplexity}
    mv = MPSE(D,visualization_method=['mds','tsne'],
              visualization_args=va,
              colors=data['colors'],verbose=2)

    mv.gd()
        
    mv.plot_computations()
    mv.plot_embedding(title='final embeding')
    mv.plot_images()
    plt.draw()
    plt.pause(0.2)
    plt.show()
    
    return

if __name__=='__main__':
    mpse_tsne('clusters', n_clusters=[2,5],
              evaluate=True, n_perspectives=7,
              verbose=2,show_plots=True)

    
    estimate_cost=False
    evaluate=True
    run_all_mpse_tsne = True
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
        mpse_tsne('123', n_samples=500, perplexity = 460,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('mnist',n_samples=500,perplexity=30,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('mnist',n_samples=500,perplexity=100,
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)
        mpse_tsne('phishing',
                  estimate_cost=estimate_cost,evaluate=evaluate,
                  verbose=2,show_plots=True)

    run_all_compare_tsne = True
    if run_all_compare_tsne is True:
        compare_tsne('clusters', n_samples=400, n_perspectives=2,
                     evaluate=True)
        
    run_all_compare_perplexity = True
    if run_all_compare_perplexity is True:
        compare_perplexity('clusters2', n_samples=400, perplexities=[10,200])
    
