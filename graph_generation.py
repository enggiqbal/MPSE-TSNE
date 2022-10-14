import graph_tool.all as gt
import numpy as np

G = gt.Graph(directed=False)
G.add_vertex(n=8)

edgeList = [
            (0,1),
            (0,2),
            (0,3),
            (1,2),
            (1,3),
            (2,3),
            (4,5),
            (4,6),
            (4,7),
            (5,6),
            (5,7),
            (6,7),
            (0,4)
            ]
G.add_edge_list(edgeList)

M = gt.shortest_distance(G)

d1 = np.array([v for v in M])

G = gt.Graph(directed=False)
G.add_vertex(n=8)

edgeList = [
            (0,5),
            (0,4),
            (0,3),
            (1,2),
            (1,7),
            (2,7),
            (4,5),
            (4,0),
            (4,3),
            (5,0),
            (5,3),
            (6,7),
            (0,2)
            ]
G.add_edge_list(edgeList)

M = gt.shortest_distance(G)

d2 = np.array([v for v in M])

labels = [str(v) for v in G.iter_vertices()]

import mview

mview.mpse_tsne( [d1,d2] ,
                perplexity=40,
                show_plots=True, verbose=2)#,sample_labels=labels)
