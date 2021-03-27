import sys
sys.path.insert(0,"../")
import mview
import numpy as np 
#two clusters in 2 perspectives	200-2000 step 200	0.6*(# points)	2
import pandas as pd 
from tqdm import tqdm

class Exp:
    def __init__(self):
        pass
    def get_error(self, results):
        errors=[]
        for i in range(results.n_perspectives):
            Y = results.images[i]
            labels = results.image_classes[i]
            error = mview.evaluate.separation_error(Y,labels )
            errors.append(error)
        return np.average(errors)
    
    def exp1(self, filename, n_perspectives=2 ):
        start=20
        end=200
        step=20
        repeat=5
        n_clusters=2
        final_results=[]
        for n_sample in tqdm(range( start, end, step)):
            for i in range (repeat):
                results = mview.mpse_tsne('clusters', n_samples=n_sample, n_clusters=n_clusters, n_perspectives=n_perspectives, perplexity=int(n_sample * 0.6),
                                            show_plots=False, verbose=0)
                e=self.get_error(results)
                final_results.append([n_sample,i,n_perspectives,n_clusters,e, results.time ])
                print(final_results[-1])
        df=pd.DataFrame(final_results, columns=['n_samples','exp_num','n_perspectives','n_clusters','separation_error','time'])
        df.to_csv(filename, index=False)


if __name__=="__main__":
    e=Exp()
    e.exp1("two_clusters_2.csv")