import numpy as np 
import pylab as plt 
import mview
from new_enstsne2 import get_clusters_distance, load_penguins
from eval import vis_3d, vis_2d_per_proj

dists, labels, X = get_clusters_distance(400,[4,3,2])
dists, labels, X = load_penguins()
mv = mview.basic(dists, verbose=2, smart_initialize=True, max_iter=1000)

vis_2d_per_proj(mv.images, labels)
vis_3d(mv.embedding, labels)
