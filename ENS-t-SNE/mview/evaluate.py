from sklearn.linear_model import LogisticRegression
def separation_error(X, y, return_individual_errors=False,
                     show_plot=False):
    clf = LogisticRegression(random_state=0).fit(X, y)
    return 1- clf.score(X,y)

if __name__=='__main__':
    import samples
    coordinates, data = samples.sload('clusters2', n_clusters=3, n_samples=40)
    yLabels = data['sample_classes']
    errorVal = separation_error(coordinates, yLabels, show_plot=True)