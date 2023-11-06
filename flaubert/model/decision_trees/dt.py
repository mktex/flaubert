import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from time import time
from sklearn.model_selection import GridSearchCV
from functools import reduce

from flaubert.statistik import stat_op
from flaubert.eda import dfgestalt
from flaubert import einstellungen

# print ohne gedöns
np.set_printoptions(suppress=True)

features_train = None
features_test = None
labels_train = None
labels_test = None
all_features = None

xlabel_basis = None

DO_OUTPUT = "python"

data_ordner = './data'

dt_datei_output = f'{data_ordner}/testout.dot'.replace("//", "/")
output_py_code_datei = f'{data_ordner}/xmodell_dt_generiert.py'.replace("//", "/")
xres_code = None

vis_conf_mat = False

def set_data(ftr, fts, ltr, lts, allf):
    global features_train, features_test, labels_train, labels_test, all_features
    features_train = ftr
    features_test = fts
    labels_train = ltr
    labels_test = lts
    all_features = allf


clf = None
clf_optim = None


def do_model(max_depth=10, min_samples_split=7, min_samples_leaf=10,
             criterion='entropy',
             max_features=24, print_statistics=True):
    global clf
    global vis_conf_mat
    clf = tree.DecisionTreeClassifier(max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      criterion=criterion,
                                      max_features=max_features,
                                      random_state=42
                                      )
    clf.fit(features_train, labels_train)
    if print_statistics:
        stat_op.print_clf_statistics(clf, features_train, labels_train,
                                     features_test, labels_test,
                                     vis_confusion_mat=vis_conf_mat)
    return clf


def do_model_optim():
    global clf, clf_optim
    global labels_train, features_train
    global vis_conf_mat
    print("Fitting the classifier to the training set")
    t0 = time()
    # min sample splitmaximum 30% of min of len labels_train and test
    # xmax_sample_split = int(len(labels_train) * 0.3)
    param_grid = {
        'max_depth': list(range(3, 6)),
        'min_samples_split': pd.Series(list(range(10, 100))).sample(10).values.tolist(),
        'max_features': [None] + list(range(5, features_train.shape[1])),
        'criterion': ['entropy', 'gini']
    }
    # for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
    clf_local = GridSearchCV(
        tree.DecisionTreeClassifier(
            max_depth='max_depth',
            min_samples_split='min_samples_split',
            criterion='criterion',
            max_features='max_features'
            # random_state=42
        ),
        param_grid,
        scoring='roc_auc'
    )
    clf_local = clf_local.fit(features_train, labels_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf_local.best_estimator_)
    clf_optim = clf_local.best_estimator_
    stat_op.print_clf_statistics(clf_optim, features_train,
                                 labels_train, features_test, labels_test,
                                 vis_confusion_mat=vis_conf_mat)


def extract_alphanumeric(InputString):
    from string import ascii_letters, digits
    return "".join([ch for ch in InputString if ch in (ascii_letters + digits)])


def toVar(xstr):
    xstr = str(xstr)
    xstr = extract_alphanumeric(xstr)
    return xstr if xstr[0].isalpha() else 'x' + xstr


def print_decision_tree(clf, feature_names=None, offset_unit='\t'):
    """
        # adapted - http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
        Plots textual representation of rules of a decision tree: this is scikit-learn representation of tree
        feature_names: list of feature names. They are set to f1,f2,f3,... if not specified
        offset_unit: a string of offset of the conditional block
    """
    global xres
    left = clf.tree_.children_left
    right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    value = clf.tree_.value
    xres = ''
    if feature_names is None:
        features = ['f%d' % i for i in clf.tree_.feature]
    else:
        features = [feature_names[i] for i in clf.tree_.feature]

    def recurse_python(left, right, threshold, features, node, depth=0):
        global xres
        offset = offset_unit * depth
        if (threshold[node] != -2):
            xres += offset + "if " + toVar(features[node]) + " <= " + str(round(threshold[node], 4)) + ":" + "\n"
            if left[node] != -1:
                recurse_python(left, right, threshold, features, left[node], depth + 1)
            xres += offset + "else:" + "\n"
            if right[node] != -1:
                recurse_python(left, right, threshold, features, right[node], depth + 1)
            xres += offset + "\t" + "\n"
        else:
            xstr = str(value[node]).replace('.', '., ')
            xstr = ''.join(xstr.split())
            xstr = xstr.replace(".,", ", ")
            xres += offset + "xresult = " + xstr + "\n"

    def recurse_sql(left, right, threshold, features, node, depth=0, output="wert", xstr_transfer=[]):
        global xres
        from copy import deepcopy
        offset = offset_unit * depth
        if (threshold[node] != -2):
            xt = deepcopy(xstr_transfer)
            xt.append(offset + ("and" if depth != 0 else "when") + " a." + toVar(features[node]) + " <= " + str(
                round(threshold[node], 4)) + "\n")
            if left[node] != -1:
                recurse_sql(left, right, threshold, features, left[node], depth + 1, output=output, xstr_transfer=xt)
            # xres += offset + "ELSE" + "\n"
            xt = deepcopy(xstr_transfer)
            xt.append(offset + ("and" if depth != 0 else "when") + " a." + toVar(features[node]) + " > " + str(
                round(threshold[node], 4)) + "\n")
            if right[node] != -1:
                recurse_sql(left, right, threshold, features, right[node], depth + 1, output=output, xstr_transfer=xt)
            # xres += offset + "END\t" + "\n"
        else:
            if output == "wert":
                xstr = str(value[node]).replace('.', '., ')
                xstr = ''.join(xstr.split())
                xstr = xstr.replace(".,", ", ")
                xexpr = eval(xstr);
                xexpr = xexpr[0]
                xstr = str(xexpr[1]) + str("/") + str(xexpr[0] + xexpr[1])
                xstr = offset + "then " + ' '.join(xstr.split())
                xstr = ' '.join(xstr_transfer + [xstr])
            elif output == "id":
                xstr = offset + "then " + str(node) + "\n"
                xstr = ' '.join(xstr_transfer + [xstr])
            xres += "\n" + xstr + "\n"

    if DO_OUTPUT == "sql":
        recurse_sql(left, right, threshold, features, 0, 0, output="wert")
        xres = xres[:-1]
        xres += "\nend\t as prob_reseller, \n"

        xres_1 = "CASE \n" + xres

        xres = ''
        recurse_sql(left, right, threshold, features, 0, 0, output="id")
        xres = xres[:-1]
        xres += "end as node_id\n"
        xres_2 = xres

        xres = xres_1 + "\n\nCASE\n" + xres_2

    elif DO_OUTPUT == "python":
        recurse_python(left, right, threshold, features, 0, 0)

    return xres


def dt_export_code(dict_clf, dt_features, dt_model_pfad):
    for nbin in dict_clf.keys():
        quellcode = dict_clf[nbin]
        model_bin = f"prevclose_bin_{nbin}"
        dt_model_pfad = f"./quellcodegen/{model_bin}.py"
        write_out(quellcode, feature_names=dt_features, dt_model_pfad=dt_model_pfad)
    """
    model_bin = f"xbin_{ist_nbin}"
    dtm = importlib.import_module(".", package=f"quellcodegen.{model_bin}")
    reload(dtm)
    prognosen = xdata[dt_features].fillna(value=xdata[dt_features].mean()) \
                                  .apply(lambda datensatz: dtm.xr(*datensatz),
                                         axis=1)    
    """


def get_dt_code(clf, feature_names=None, offset_unit='\t'):
    global xres

    left = clf.tree_.children_left
    right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    xres = ''
    if feature_names is None:
        features = ['f%d' % i for i in clf.tree_.feature]
    else:
        features = [feature_names[i] for i in clf.tree_.feature]

    def recurse_python(left, right, threshold, features, node, depth=0, dt_pfad=""):
        global xres
        offset = offset_unit * depth
        if threshold[node] != -2:
            bedingung = toVar(features[node]) + " <= " + str(round(threshold[node], 4))
            xres += offset + "if " + bedingung + ":" + "\n"
            if left[node] != -1:
                recurse_python(left, right, threshold, features, left[node], depth + 1,
                               dt_pfad + " # " + bedingung)
            xres += offset + "else:" + "\n"
            if right[node] != -1:
                recurse_python(left, right, threshold, features, right[node], depth + 1,
                               dt_pfad + " # " + bedingung.replace("<=", ">"))
            xres += offset + "\t" + "\n"
        else:
            xstr = str(value[node]).replace('.', '., ')
            xstr = ''.join(xstr.split())
            xstr = xstr.replace(".,", ", ")
            xres += offset + f"xresult = [{xstr}, '{dt_pfad}']\n"

    recurse_python(left, right, threshold, features, 0, 0)

    return xres


def write_out(xres, feature_names=None, dt_model_pfad=None):
    global output_py_code_datei
    if dt_model_pfad is None: dt_model_pfad = output_py_code_datei
    if feature_names is None:
        feature_names = reduce(lambda a, b: a + ', ' + b, ['f' + str(x) for x in range(len(feature_names))])
    feature_names_str = ", ".join([toVar(x.replace("_", "")) for x in feature_names])
    xr = 'def xr(' + feature_names_str + '):\n\n'
    xr += reduce(lambda a, b: a + "\n" + b,
                 ['\t' + y for y in [x for x in xres.split('\n') if x.strip() != '']]
    )
    xr += "\n\treturn xresult"
    with open(dt_model_pfad, 'w') as f:
        f.write(xr)


def export_dt_vis(xfeature_names, fname_dt="tree_model_all_data"):
    from sklearn import tree
    # from six import StringIO
    # import pydot
    import os
    global clf, all_features
    global xres_code, dt_datei_output
    global data_ordner
    global output_py_code_datei
    xclass_names = list(map(lambda x: str(x), clf.classes_))
    fig = plt.Figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_HOCH)
    _ = tree.plot_tree(clf,
                       feature_names=all_features,
                       class_names=xclass_names,
                       filled=True, rounded=True)

    fig.savefig(f'{data_ordner}/{fname_dt}.png', dpi=einstellungen.FIGDPI_NORM)
    """
    # Variante 1:
    f = open(dt_datei_output, 'w')
    tree.export_graphviz(clf, out_file=f,
                         feature_names=all_features,
                         class_names=xclass_names,
                         filled=True, rounded=True
    )
    f.close()
    # diese Variante ist farbiger..
    os.system(f"dot -Tpng {data_ordner}/testout.dot -o {data_ordner}/{fname_dt}.png")    
    
    # Variante 2:
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, node_ids=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png(f"{data_ordner}/dtvis.png")
    """
    output_py_code_datei = f'{data_ordner}/xmodel_{fname_dt}_generiert.py'.replace("//", "/")
    xres = print_decision_tree(clf, feature_names=xfeature_names)
    xres_code = xres
    write_out(xres, xfeature_names)


def warum(DCLUSTERInput, xcolumnsSet, xcolTarget, xlambdaF, useOrdered=None, balancedN=None,
          test_size=0.3, max_depth=5, min_samples_split=25, min_samples_leaf=25, criterion="entropy",
          print_stats=True, print_dtstruktur=False, dt_visualisierung=False, fname_dt=None):
    """
    reload(dt)
    dt.warum(xdata, xcolumnsSet=filter(lambda x: x != "target", xdata.columns), xcolTarget="target",
              xlambdaF=lambda x: 'g0' if x < 5.0 else 'g1',
              useOrdered=["g0", "g1"], balancedN={"g1":480, "g0":1500},
              test_size=0.2, max_depth=2, min_samples_split=25, min_samples_leaf=25, criterion="entropy")
    """
    global clf
    from sklearn.model_selection import train_test_split
    X, y, dtFrame, xlabels, xlabels_basis = dfgestalt.split_data_groups(DCLUSTERInput, xcolumnsSet, xcolTarget,
                                                                        xlambdaF, useOrdered, balancedN)
    feature_names = dtFrame.columns.values
    X_train, X_test, y_train, y_test = train_test_split(X, xlabels, test_size=test_size)  # , random_state=42
    set_data(X_train, X_test, y_train, y_test, feature_names)
    clf = do_model(max_depth=max_depth,
                   min_samples_split=min_samples_split,
                   min_samples_leaf=min_samples_leaf,
                   criterion=criterion,
                   max_features=dtFrame.shape[1],
                   print_statistics=print_stats
    )
    if print_dtstruktur: print("\n", print_decision_tree(clf, feature_names=feature_names), "\n")
    if dt_visualisierung and fname_dt is not None: export_dt_vis(feature_names, fname_dt)
    return clf
