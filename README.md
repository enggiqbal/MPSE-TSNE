**Embedding Neighborhoods Simultaneously by t-Distributed Stochastic Neighbor Embeding: ENS-t-SNE**

Please see the following video for a demonstration and explanation: [![ENS-t-SNE Demonstration Video](https://img.youtube.com/vi/kRFcNs29ttA/0.jpg)](https://www.youtube.com/watch?v=kRFcNs29ttA)

An example program and visualization script can be found in ```enstsne.py```

To import the ens-t-sne module, use 

```
import mview
```

To run ens-t-sne and store results in an object ```mv```

```
mv = mview.mpse_tsne(
            data, 
            perplexity,
            iters,
            smart_init,
            verbose,
            show_plots, 
            save_results)

```

```data``` is a list of either symmetric numpy distance matrices or nxd numpy high dimensional feature array. 

```perplexity``` is the perplexity parameter for the t-SNE optimizations, defaults to 30. 

```iters``` is the total number of updates to run in the optimization, defaults to 200. 

```smart_init``` is whether to use a smart initialization strategy (True) or to randomly initialize the optimization (False), defaults to True. 

```verbose``` is the verbosity level, defaults to 0.

```show_plots``` is whether to plot results and call plt.show() after optimization, defaults to False 

```save_results``` is whether to store the various arrays produced by the algorithm into csv files. Defaults to False. If True, saves results to the ```mview/temp``` directory by default, but can be changed by passing a string ```output="path/to/directory"``` with a path to the desired destination. 

For a full list of optional keyword arguments, see the documentation in the ```mview/mpse.py``` file.

-----------------------------------------------------

The ```mview.mpse_tsne``` function returns an object with the following properties: 

```mpse_tsne.embedding``` is an nx3 array containing the 3-dimensional embedding of the dataset. 

```mpse_tsne.images``` is an mxnx2 array, the first index corresponding to the projection number, the second to each datapoint, and the third to it's x,y coordinates

```mpse_tsne.projections``` is an mx3x2 array, with the first index corresponding to the projection number, and the remaining two being the 3x2 projection matrix. 