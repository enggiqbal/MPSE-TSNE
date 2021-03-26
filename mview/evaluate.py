import matplotlib.pyplot as plt
import numpy as np

def separation_error(X, y, plot=False):
    "finds best linear separator and then computes overall error"
    import sklearn
    from sklearn import svm
    from sklearn.datasets import make_blobs
    
    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    predicted = clf.predict(X)
    
    if plot is True:
        ## uncomment this to show the plots
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        # plot the decision function
        ax = plt.gca()
        print(ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ##create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        ## plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        plt.show()

    return min(np.linalg.norm([1]*len(predicted) - predicted - y),
               np.linalg.norm(predicted - y))
