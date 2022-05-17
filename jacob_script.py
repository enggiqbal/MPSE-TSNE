import mview
import graph_tool.all as gt
import numpy as np

mview.mpse_tsne('florence',
                perplexity=10,
                show_plots=True, verbose=2)


# G1 = gt.load_graph('small_cluster.dot')
# d1 = np.array([ v for v in gt.shortest_distance(G1) ])
# e1 = [ [u,v] for u,v in G1.iter_edges() ]
#
# G2 = gt.load_graph('small_cluster2.dot')
# d2 = np.array([ v for v in gt.shortest_distance(G2) ])
# e2 = [ [u,v] for u,v in G2.iter_edges() ]
# e1 = e1 + e2
#
# kwargs = {'edges': [e1,e1], 'sample_labels': [str(v) for v in G1.iter_vertices()]}
#
# mview.mpse_tsne([d1,d2], perplexity=2,show_plots=True,verbose=2,**kwargs)
