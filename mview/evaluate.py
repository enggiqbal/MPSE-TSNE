import matplotlib.pyplot as plt
import numpy as np

def separation_error0(X, y, show_plot=False):
    "finds best linear separator and then computes overall error"
    import sklearn
    from sklearn import svm
    from sklearn.datasets import make_blobs
    
    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    predicted = clf.predict(X)
    
    if show_plot is True:
        ## uncomment this to show the plots
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        # plot the decision function
        ax = plt.gca()
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
        plt.draw()

    error = min(np.linalg.norm([1]*len(predicted) - predicted - y),
                np.linalg.norm(predicted - y))
    return error**2/len(X)

def separation_error(coordinates, labels, return_individual_errors=False,
                     show_plot=False):
    "returns all separation errors"
    labels = np.array(labels)
    errors = []
    n_samples = len(coordinates)
    n_classes = np.max(labels)+1
    assert n_classes > 1
    for i in range(n_classes):
        for j in range(i+1,n_classes):
            indices = [k for k in range(n_samples) if labels[k] in [i,j]]
            X = coordinates[indices]
            y = labels[indices]
            errors.append(separation_error0(X,y,show_plot))
    if show_plot:
        plt.show()
    if return_individual_errors:
        return errors
    else:
        return np.mean(errors)

    # we create 40 separable points
if __name__=='__main__':
    import samples
    coordinates, data = samples.sload('clusters2', n_clusters=3, n_samples=40)
    yLabels = data['sample_classes']
    errorVal = separation_error(coordinates, yLabels, show_plot=True)


