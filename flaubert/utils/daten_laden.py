import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flaubert.transform import transform_norm as tnorm
from flaubert.transform import kateg


def lade_zufalldatenbestand(dim_x, dim_y, n_classes, pinf=0.7, n_samples=1000, class_sep=1.3,
                            typ="clf", do_norm=True, target_var_to_one_hot_encoding=False):
    """ typ: clf | reg """
    nfeatures = dim_x * dim_y
    ninformative = int(pinf * nfeatures)
    nredundant = nfeatures - ninformative - 1
    print(f"[x] {ninformative} sind nützlich, {nredundant} sind redundant")
    X_mins, X_max = None, None
    if typ == "clf":
        X, Y = datasets.make_classification(n_samples=n_samples,
                                            n_classes=n_classes,
                                            n_features=nfeatures,
                                            n_informative=ninformative,
                                            n_redundant=nredundant,
                                            n_repeated=1, class_sep=class_sep)
    elif typ == "reg":
        X, Y = datasets.make_regression(n_samples=n_samples, n_targets=1, n_features=nfeatures, n_informative=ninformative)
    if do_norm:
        X_norm, X_mins, X_max = tnorm.normalisierung_mehrdimensional(X)
    else:
        X_norm = X
    y = pd.Series(Y.flatten())
    if target_var_to_one_hot_encoding:
        Ydigits = Y
        if typ == "reg":
            ybins = [y.quantile(p) for p in np.arange(0, 1, 0.2)]
            Ydigits = np.digitize(Y.flatten(), bins=ybins)
        df = pd.DataFrame({"xlabel": ["id_{}".format(t) for t in Ydigits]})
        y_one_hot_encoded = kateg.kateg2dummy(df).values
    else:
        y_one_hot_encoded = Y
    x_train, x_test, y_train, y_test = train_test_split(X_norm, y_one_hot_encoded, test_size=0.3, random_state=None)
    return x_train, y_train, x_test, y_test, X_mins, X_max