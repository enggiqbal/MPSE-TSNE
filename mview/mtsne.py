import sys
import numbers, math, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance

import misc, setup, multigraph, gd, projections, mds, tsne, plots, mpse, samples
from mpse import MPSE

def mpse_tsne(data, perplexity=30, iters=50,
              estimate_cost=False,
              verbose=2, show_plots=True, save_results = False,**kwargs):
    "Runs MPSE optimized for tsne"
    
    #load data
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
    n_samples = mv.n_samples

    #search for global minima
    mv.gd(fixed_projections=True, max_iter=10, batch_size=min(25,n_samples//2),
          scheme='mm')
    for divisor in [20,10,5,2]:
        batch_size = max(5,min(500,n_samples//divisor))
        mv.gd(batch_size=batch_size, max_iter=20, scheme='mm')
    #mv.gd(max_iter=20, scheme='mm')
    mv.gd(max_iter=iters, scheme='fixed')
        
    #save outputs:
    if save_results is True:
        mv.save()

    if show_plots is True:
        mv.plot_computations()
        mv.plot_embedding()
        mv.plot_images(**kwargs)
        plt.show()
    return mv

def compare_perplexity(dataset='clusters', perplexities=[30,200], **kwargs):
    data = samples.sload(dataset, **kwargs)
    D = [data['D']]*2
    va = []
    for p in perplexities:
        va.append({'perplexity':p})
    mv = MPSE(D,visualization_method='tsne',
              fixed_projections='standard',
              visualization_args=va,
              colors=data['colors'],verbose=2)

    mv.optimized()
        
    #mv.plot_computations()
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
    run_all_mpse_tsne = True
    #mpse_tsne('disk2', n_samples=100, n_perspectives=3, perplexity=30,
    #          estimate_cost=False)

    #compare_perplexity(dataset='clusters2', n_samples=500, perplexities=[5,30])
    #compare_mds_tsne()

    if run_all_mpse_tsne is True:
        mpse_tsne('equidistant', save_results=True)
        mpse_tsne('disk', n_perspectives=10)
        mpse_tsne('clusters', n_clusters=[3,4,2])
        mpse_tsne('clusters2', n_clusters=2, n_perspectives=4, perplexity=80)
        mpse_tsne('florence', perplexity = 40)
        mpse_tsne('123', n_samples=500, perplexity = 980)
        mpse_tsne('credit')
        mpse_tsne('mnist',n_samples=500,perplexity=30)
        mpse_tsne('mnist',n_samples=1000,perplexity=100)
        mpse_tsne('phishing')
    
