Following funcitons can be used to analyze PCA data.

Once we fit and transform data using PCA, we need to select optimal PCA vectors to reduce the dimension

One way is to get eigen values that represents 90% (for example) of the variance.

Below functions will help visualize and select those values

```
def showPcaCumVariance(pca_model):
    '''
    Plots Cumulative Variance of given PCA model
    using its eigen values to calculate CDF
    '''
    pca_percent_explained = pca_model.explained_variance_ / np.sum(pca_model.explained_variance_)
    cum_var_explained = np.cumsum(pca_percent_explained)
    plt.figure(1, figsize=(6,4))
    plt.clf()
    plt.plot(cum_var_explained, linewidth=2)
    return

def getPcaEigenValuesCount(pca_model, req_perc=0.9):
    '''
    returns Max number of PCA eigen values needed to have @req_perc variance
    '''
    max_pca_needed = 0
    pca_percent_explained = pca_model.explained_variance_ / np.sum(pca_model.explained_variance_)
    cum_var_explained = np.cumsum(pca_percent_explained)
    
    for c in cum_var_explained:
        if c < req_perc:
            max_pca_needed += 1
    return max_pca_needed
```