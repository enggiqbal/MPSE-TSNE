import sys
sys.path.insert(0,"../")
import mview
import numpy as np 
#two clusters in 2 perspectives	200-2000 step 200	0.6*(# points)	2
import pandas as pd 
from tqdm import tqdm
import math
import time
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
    def exp0(self):
        n_clusters=2
        n_sample=60 
        for i in range (15):
            results = mview.mpse_tsne('clusters', n_samples=n_sample, n_clusters=n_clusters, 
            n_perspectives=2, perplexity=int(n_sample * 0.6),  show_plots=False, verbose=0)
            print(results.images[0][0])

    def exp1(self, filename, f_perplexity=0.2, n_perspectives=2 ):
        t=time.time()
        start=200
        end=2000
        step=200
        repeat=5
        n_clusters=2
        final_results=[]
        for n_sample in tqdm(range( start, end, step)):
            i=0
            while(i<3):
                t=time.time()
                results = mview.mpse_tsne('clusters',evaluate=True, n_samples=n_sample, n_clusters=n_clusters, n_perspectives=n_perspectives, perplexity=int(n_sample * f_perplexity),  show_plots=False, verbose=0)
      
                # import pdb; pdb.set_trace()
                if math.isnan( results.images[0][0][0]) ==False:
                    e2= results.image_separation
                    e=np.mean(e2)
                    final_results.append([n_sample,i,n_perspectives,n_clusters,e, results.time ])
                    print(final_results[-1], "time", time.time() -t )
                    i=i+1
                else:
                    print("scaping")
        df=pd.DataFrame(final_results, columns=['n_samples','exp_num','n_perspectives','n_clusters','separation_error','time'])
        df.to_csv(filename, index=False)
 
    def clusters_x_fixed_point(self,filename,n_perspectives=2):
        t=time.time()
        n_sample=400
        perplexitys=[40, 80, 160, 240]
        
        final_results=[]
        for perplexity in perplexitys:
            for n_clusters in tqdm(range( 2, 11)):
                i=0
                while(i<5):
                    t=time.time()
                    results = mview.mpse_tsne('clusters',evaluate=True,n_clusters=n_clusters, n_samples=n_sample,  n_perspectives=n_perspectives, perplexity=perplexity,  show_plots=False, verbose=0)
                    if math.isnan( results.images[0][0][0]) ==False:
                        e2= results.image_separation
                        e=np.mean(e2)
                        final_results.append([perplexity,n_sample,i,n_perspectives,n_clusters,e, results.time ])
                        print(final_results[-1], "time", time.time() -t )
                        i=i+1
                    else:
                        print("scaping")
        df=pd.DataFrame(final_results, columns=['perplexity','n_samples','exp_num','n_perspectives','n_clusters','separation_error','time'])
        df.to_csv(filename, index=False)

    def exp2(self, filename):
        t=time.time()
        n_sample=1000
   
        n_clusters=2
        final_results=[]
        for n_perspectives in tqdm(range( 2, 11)):
            i=0
            while(i<3):
                t=time.time()
                results = mview.mpse_tsne('disk2',evaluate=True, n_samples=n_sample,  n_perspectives=n_perspectives, perplexity=int(n_sample * 0.6),  show_plots=False, verbose=0)
                if math.isnan( results.images[0][0][0]) ==False:
                    e2= results.image_separation
                    e=np.mean(e2)
                    final_results.append([n_sample,i,n_perspectives,n_clusters,e, results.time ])
                    print(final_results[-1], "time", time.time() -t )
                    i=i+1
                else:
                    print("scaping")
        df=pd.DataFrame(final_results, columns=['n_samples','exp_num','n_perspectives','n_clusters','separation_error','time'])
        df.to_csv(filename, index=False)
    def exp3(self, filename):
        n_sample=400
        n_clusters=2
        perplexitys=[40, 80, 160, 240]
        final_results=[]
        for perplexity in perplexitys:
            for n_perspectives in tqdm(range( 2, 11)):
                i=0
                while(i<5):
                    t=time.time()
                    results = mview.mpse_tsne('clusters',evaluate=True, n_samples=n_sample,  n_perspectives=n_perspectives, perplexity=perplexity,  show_plots=False, verbose=0)
                    if math.isnan( results.images[0][0][0]) ==False:
                        e2= results.image_separation
                        e=np.mean(e2)
                        final_results.append([n_sample,i,n_perspectives,n_clusters,perplexity,e, results.time ])
                        print(final_results[-1], "time", time.time() -t )
                        i=i+1
                    else:
                        print("scaping")
        df=pd.DataFrame(final_results, columns=['n_samples','exp_num','n_perspectives','n_clusters','perplexity','separation_error','time'])
        df.to_csv(filename, index=False)
if __name__=="__main__":
    e=Exp()
    # e.exp1("two_clusters_2_0.6.csv",0.6,2)
    # e.exp1("two_clusters_2_0.2.csv",0.2,2)
    # e.exp1("two_clusters_3_0.2.csv",0.2,3)
    # e.exp2("disk_0.6.csv")
    # e.exp3("perspectives2.csv")
    e.clusters_x_fixed_point("clusters_2_fixed_point.csv")
    e.clusters_x_fixed_point("clusters_3_fixed_point.csv",3)

 