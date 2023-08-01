from sklearn.neighbors import LocalOutlierFactor
import pandas as pd


def ausreisser_lof(xs, n_neighbors=30):
    """ Ergibt ein Pandas Series mit entsprechenden LOF Markierung """
    clf = LocalOutlierFactor(n_neighbors)
    xindex = xs.index
    xdf = pd.DataFrame({'werte': xs})
    res = pd.DataFrame({
        "neg_LOF": clf.fit_predict(xdf)
    })
    res.index = xindex
    return res.dropna()
