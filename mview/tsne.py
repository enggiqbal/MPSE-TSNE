### tSNE implementation ###
import numbers, math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial.distance
from scipy.spatial.distance import squareform
MACHINE_EPSILON = np.finfo(np.double).eps

import misc, gd, plots, setup, evaluate

### joint probabilities from distances ###

def joint_probabilities(distances, perplexity):
    """\
    Computes the joint probabilities p_ij from given distances (see tsne paper).

    Parameters
    ----------

    distances : array, shape (n_samples*(n_samples-1)/2,)
    Pairwise distances, given as a condensed 1D array.

    perpelxity : float, >0
    Desired perplexity of the joint probability distribution.
    
    Returns
    -------

    P : array, shape (n_samples*(n_samples-1)/2),)
    Joint probability matrix, given as a condensed 1D array.
    """
    #change condensed distance array to square form
    distances = scipy.spatial.distance.squareform(distances)
    n_samples = len(distances)
    
    #find optimal neighborhood parameters to achieve desired perplexity
    lower_bound=1e-1; upper_bound=1e1; iters=10 #parameters for binary search
    sigma = np.empty(n_samples) #bandwith array
    for i in range(n_samples):
        D_i = np.delete(distances[i],i) #distances to ith sample
        estimate = np.sum(D_i)/(n_samples-1)/5
        lower_bound_i=lower_bound*estimate; upper_bound_i=upper_bound*estimate;
        for iter in range(iters):
            #initialize bandwith parameter for sample i:
            sigma_i = (lower_bound_i*upper_bound_i)**(1/2)
            #compute array with conditional probabilities w.r.t. sample i:
            P_i = np.exp(-D_i**2/(2*sigma_i**2))
            if np.isfinite(P_i).all() is False:
                print('infinite value')
            if np.nan in P_i:
                print('nan found')
            if np.sum(P_i) == 0:
                print('adds to 0')
            P_i /= np.sum(P_i)
            #compute perplexity w.r.t sample i:
            HP_i = -np.dot(P_i,np.log2(P_i+MACHINE_EPSILON))
            PerpP_i = 2**(HP_i)
            #update bandwith parameter for sample i:
            if PerpP_i > perplexity:
                upper_bound_i = sigma_i
            else:
                lower_bound_i = sigma_i
        #final bandwith parameter for sample i:
        sigma[i] = (lower_bound_i*upper_bound_i)**(1/2)

    #compute conditional joint probabilities (note: these are transposed)
    conditional_P = np.exp(-distances**2/(2*sigma**2))
    np.fill_diagonal(conditional_P,0)
    conditional_P /= np.sum(conditional_P,axis=0)

    #compute (symmetric) joint probabilities
    P = (conditional_P + conditional_P.T)
    P = scipy.spatial.distance.squareform(P, checks=False)
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(P/sum_P, MACHINE_EPSILON)

    return P

### Cost function and gradient ###

def inverse_square_law_distances(embedding):
    """\
    Computes the pairwise inverse square law distances for the given embedding. 
    These are given by q_ij = 1/(1+|y_i-y_j|^2) for a given pair (i,j). The set 
    of probabilities used in the low dimensional map of tSNE can then be computed
    by dividing by the sum.

    Parameters
    ----------

    embedding : array, shape (n_samples,dim)
    Embedding (coordinates in low-dimensional map).

    Returns
    dist: pairwise inverse square law distances, as a condensed 1D array.
    """
    dist = scipy.spatial.distance.pdist(embedding,metric='sqeuclidean')
    dist += 1.0
    dist **= -1.0
    return dist
    
def KL(P,embedding):
    """\
    KL divergence KL(P||Q) between distributions P and Q, where Q is computed
    from the student-t distribution from the given embedding array.

    Parameters
    ----------

    P : array, shape (n_samples*(n_samples-1)/2,)
    Joint probabilities, given as a condensed 1D array.
    
    embedding : array, shape (n_samples,dim)
    Current embedding.

    Returns
    -------

    kl_divergence : float
    KL-divergence KL(P||Q).
    """
    # compute Q matrix
    dist = inverse_square_law_distances(embedding)
    Q = np.maximum(dist/(np.sum(dist)), MACHINE_EPSILON)

    # compute kl divergence
    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P/Q, MACHINE_EPSILON)))
        
    return kl_divergence

def grad_KL(P,embedding,dist=None,Q=None):
    """\
    Computes KL divergence and its gradient at the given embedding.

    Parameters
    ----------

    P : array, shape (n_samples*(n_samples-1)/2,)
    Condensed probability array.
    
    embedding : array, shape (n_samples,dim)
    Current embedding.

    Q : array, shape (n_samples*(n_samples-1)/2,)
    Joint probabilities q_ij. If not included, these are computed.

    Results
    -------

    kl_divergence : float
    KL-divergence KL(P||Q).

    grad : float
    gradiet of KL(P||Q(X)) w.r.t. X.
    """
    if dist is None or Q is None:
        dist = inverse_square_law_distances(embedding)
        Q = np.maximum(dist/(np.sum(dist)), MACHINE_EPSILON)
    
    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P/Q, MACHINE_EPSILON)))

    grad = np.ndarray(embedding.shape)
    PQd = scipy.spatial.distance.squareform((P-Q)*dist)
    for i in range(len(embedding)):
        grad[i] = np.dot(np.ravel(PQd[i],order='K'),embedding[i]-embedding)
    grad *= 4
    
    return grad, kl_divergence

def batch_gradient(P, embedding, batch_size=10, indices=None,
                   estimate_cost=True):
    """\
    Returns gradient approximation.
    """
    n_samples = len(embedding)
    if indices is None:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
    else:
        assert len(indices) == n_samples
        
    grad = np.empty(embedding.shape)
    stress = 0
    batch_number = n_samples//batch_size
    start = 0; end = start+batch_size
    for i in range(batch_number):
        if i < n_samples%batch_number:
            end +=1
        batch_idx = np.sort(indices[start:end])
        embedding_batch = embedding[batch_idx]
        P_batch = P[setup.batch_indices(batch_idx,n_samples)]
        dist = inverse_square_law_distances(embedding_batch)
        Q_batch = dist/(np.sum(dist))/(n_samples/len(batch_idx))**2
        grad0, st0 = grad_KL(P_batch,embedding_batch,
                                       dist=dist,Q=Q_batch)
        grad[batch_idx] = grad0*n_samples/(end-start)
        stress += st0*n_samples/(end-start)
        start = end
        end = start + batch_size
    if estimate_cost is False:
        stress = KL(P, embedding)
    return grad, stress

class TSNE(object):
    """\
    Class to solve tsne problems
    """
    def __init__(self, data, dim=2, perplexity=30.0, estimate_cost=True,
                 sample_labels=None, sample_classes=None, sample_colors=None,
                 edges=None, verbose=0, indent='', **kwargs):
        """\
        Initializes TSNE object.

        Parameters
        ----------

        data : array or dictionary
        Contains distances/dissimilarities among a set of objects.
        Can be either a square distance matrix, a condence distance array, or
        a feature array (in which case the distances are computed using the
        euclidean distance, or another specified metric).

        dim : int > 0
        Embedding dimension.

        perplexity : float > 0
        Perplexity used in determining the conditional probabilities p(i|j).

        estimate_cost : booelan
        If set to True (default), when a batch gradient descent is computed, the
        current cost will only be estimated (which requires no extra 
        computations).

        sample_labels, sample_classes, sample_colors : list
        Labels, classes, and colors for each sample. Used in plots.

        verbose : int >= 0
        Print status of methods in MDS object if verbose > 0.

        indent : str
        When printing, add indent before printing every new line.
        """
        #verbose
        self.verbose = verbose
        self.indent = indent
        if self.verbose > 0:
            print(self.indent+'mview.TSNE():')

        #sample labels, classes, and colors
        self.sample_labels = sample_labels
        self.sample_classes = sample_classes
        self.sample_colors = sample_colors
        self.edges = edges

        #setup distances and sample size
        self.distances = setup.setup_distances(data, **kwargs)
        self.n_samples = scipy.spatial.distance.num_obs_y(self.distances)
        assert isinstance(dim,int); assert dim > 0
        self.dim = dim

        #remove when safe to do so: #########################################
        self.N = self.n_samples
        self.D = self.distances

        if verbose > 0:
            print(indent+'  data details:')
            print(indent+f'    number of samples : {self.n_samples}')
            print(indent+'  embedding details:')
            print(indent+f'    embedding dimension : {dim}')
            print(indent+f'    perplexity : {perplexity:0.2f}')

        #compute joint probabilities
        self.perplexity = perplexity
        self.P = joint_probabilities(self.distances,perplexity)

        #define objective and gradient functions
        self.objective = lambda X, P=self.P, **kwargs : KL(P,X)
        def gradient(embedding, batch_size=None, indices=None, **kwargs):
            if batch_size is None or batch_size >= self.n_samples:
                return grad_KL(self.P,embedding)
            else:
                return batch_gradient(self.P, embedding, batch_size, indices,
                                      estimate_cost=estimate_cost)
        self.gradient = gradient

        #list with computation history
        self.computation_history = []

        #setup initial embedding and cost
        self.initialize(**kwargs)

    def initialize(self, initial_embedding=None, **kwargs):
        """\
        Set initial embedding.
        """
        if self.verbose > 0:
            print(self.indent+'  TSNE.initialize():')
            
        X0 = initial_embedding
        if X0 is None:
            X0 = misc.initial_embedding(self.n_samples, dim=self.dim,
                                        radius=1, **kwargs)
            if self.verbose > 0:
                print(self.indent+'    method : random')
        else:
            assert isinstance(X0,np.ndarray)
            assert X0.shape == (self.n_samples,self.dim)
            if self.verbose > 0:
                print(self.indent+'    method : initialization given')
            
        self.update(X0)
        self.initial_embedding = X0
        
        if self.verbose > 0:
            print(self.indent+f'    initial cost : {self.cost:0.2e}')
        
    def update(self,X,H=None):
        "update embedding, cost, computation history"
        self.embedding = X
        self.cost = self.objective(self.embedding)
        if H is not None:
            self.computation_history.append(H)   

    def gd(self, batch_size=None, lr=None, **kwargs):
        "run gradient descent on current embedding"
        if self.verbose > 0:
            print(self.indent+'  TSNE.gd():')
            print(self.indent+'    specs:')

        if lr is None:
            if len(self.computation_history) != 0:
                lr = self.computation_history[-1]['lr']
            else:
                lr = 100

        if batch_size is None or batch_size<2 or batch_size>self.n_samples/2:
            Xi = None
            F = lambda embedding : self.gradient(embedding)
            if self.verbose > 0:
                print(self.indent+'      gradient type : full')
        else:
            def Xi():
                indices = np.arange(self.n_samples)
                np.random.shuffle(indices)
                xi = {
                    'indices' : indices
                }
                return xi
            F = lambda X, indices : self.gradient(X, batch_size, indices)
            if self.verbose > 0:
                print(self.indent+'      gradient type : batch')
                print(self.indent+'      batch size :', batch_size)

        X, H = gd.single(self.embedding, F, Xi=Xi, lr=lr,
                        verbose=self.verbose,
                        indent=self.indent+'    ',
                        **kwargs)
        self.update(X,H)
        if self.verbose > 0:
            print(self.indent+f'    final stress : {self.cost:0.2e}')

    def optimized(self, iters=50, **kwargs):
        "attempts to find best embedding"
        if self.verbose > 0:
            print(self.indent+'  TSNE.optimized():')
            self.indent+='  '

        for divisor in [20,10,5,2]:
            batch_size = max(5,min(500,self.n_samples//divisor))
            self.gd(batch_size=batch_size, max_iter=20, scheme='mm')
        self.gd(max_iter=iters,scheme='fixed')
        if self.verbose >0:
            self.indent = self.indent[0:-2]

    def evaluate(self):
        if self.sample_classes is not None:
            Y = self.embedding; labels = self.sample_classes
            sep = evaluate.separation_error(Y, labels)
            self.separation = sep
        else:
            self.separation = None
            
    def plot_embedding(self,title='', edges=True, colors=True, labels=None,
                       axis=True, plot=True, ax=None, **kwargs):
        assert self.dim >= 2
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False
        if edges is True:
            edges = self.edges
        if colors is True:
            colors = self.sample_colors
        if isinstance(colors, int):
            assert colors in range(self.n_samples)
            colors = squareform(self.distances)[colors]            
        plots.plot2D(self.embedding,edges=edges,colors=colors,labels=labels,
                     axis=axis,ax=ax,title=title,**kwargs)
        if plot is True:
            plt.draw()
            plt.pause(1)

    def plot_computations(self, title='computations', plot=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plot = False
        costs = np.array([])
        grads = np.array([])
        lrs = np.array([])
        steps = np.array([])
        iterations=0; markers = []
        for H in self.computation_history:
            if iterations != 0:
                ax.axvline(x=iterations-1,ls='--',c='black',lw=.5)
            iterations += H['iterations']
            costs = np.concatenate((costs,H['costs']))
            grads = np.concatenate((grads,H['grads']))
            lrs = np.concatenate((lrs,H['lrs']))
            steps = np.concatenate((steps,H['steps']))
        ax.semilogy(costs,label='stress',linewidth=3)
        ax.semilogy(grads,label='grad size')
        ax.semilogy(lrs,label='lr')
        ax.semilogy(steps,label='step size')
        ax.legend()
        ax.set_title(title)
        if plot is True:
            plt.draw()
            plt.pause(0.1)

### COMPARISON ###

def sk_tsne():
    "tsne using sk-learn"

    X_true = np.load('examples/123/true2.npy')#[0:500]
    from scipy import spatial
    D = spatial.distance_matrix(X_true,X_true)
    
    from sklearn.manifold import TSNE as tsne
    X_embedded = tsne(n_components=2,
                      verbose=2,method='exact').fit_transform(X_true)
    plt.figure()
    plt.plot(X_embedded[:,0],X_embedded[:,1],'o')
    plt.show()
    
### TESTS ###

def basic(data, **kwargs):
    print()
    print('***mview.tsne.basic()***')
    print('description: a basic run of mview.TSNE on a sample dataset')
    if isinstance(data,str):
        print('dataset :',data)
    print()
    
    if isinstance(data,str):
        import samples
        kwargs0 = kwargs
        distances, kwargs = samples.sload(data, **kwargs0)
        for key, value in kwargs0.items():
            kwargs[key] = value
    else:
        distances = data
    
    vis = TSNE(distances, verbose=2, indent='  ', **kwargs)

    vis.optimized(**kwargs)
    vis.plot_computations()
    vis.plot_embedding()
    plt.show()
    
if __name__=='__main__':
    print('\n***mview.tsne : running tests***\n')

    basic('disk', dim=3, n_samples=300,
          estimate_cost=True)
    run_all_tsne=True
    estimate_cost=False
    if run_all_tsne:
        basic('equidistant', n_samples=200,
              estimate_cost=estimate_cost)
        basic('disk', n_samples=300,
              estimate_cost=estimate_cost)
        basic('disk', n_samples=300, perplexity=100,
              estimate_cost=estimate_cost)        
        basic('clusters', n_samples=200,
              estimate_cost=estimate_cost)
        basic('clusters', n_samples=400, n_clusters=8,
              estimate_cost=estimate_cost)
        basic('clusters2', n_samples=400, n_clusters=3,
              estimate_cost=estimate_cost)
        basic('mnist', n_samples=500, digits=[0,1,2,3],
              estimate_cost=estimate_cost)
