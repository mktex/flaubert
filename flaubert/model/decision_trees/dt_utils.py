import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict

import numpy as np
from flaubert.eda import dfgestalt
from flaubert.statistik import stat_op


def check_accuracy_vs_depth(features, targets, depths, ptrain=0.5, treetype="forest"):
    """ messt die Präzision dadurch dass DT viele male trainiert wird
        Problematisch hier, DT hat zwischen einzelne Training-Session keine Gedächtnis
        treeType={'forest', 'classifier', 'regressor'}
        Beispiel:
            r1, r2 = check_accuracy_vs_depth(features, targets, depths, ptrain=0.8, treetype="classifier")
    """
    train_accs = []
    test_accs = []
    for depthk in list(range(0, len(depths))):
        _ = dfgestalt.splitdata_train_test(features, targets, ptrain)
        train_features, train_targets, test_features, test_targets = _
        if depthk != 0 and (int(0.1 * train_features.shape[0]) % depthk == 0):
            print(depthk)
        sindKlassen = True
        if treetype == "forest":
            dtclf = RandomForestClassifier(max_depth=depths[depthk])
        elif treetype == "regressor":
            dtclf = DecisionTreeRegressor(max_depth=depths[depthk])
            sindKlassen = False
        else:
            dtclf = DecisionTreeClassifier(max_depth=depths[depthk])
        dtclf.fit(train_features, train_targets)
        r1 = stat_op.median_diff(dtclf.predict(train_features), train_targets, sindKlassen=sindKlassen)
        r2 = stat_op.median_diff(dtclf.predict(test_features), test_targets, sindKlassen=sindKlassen)
        train_accs.append(r1)
        test_accs.append(r2)
    plt.plot(depths, train_accs, label='Training')
    plt.plot(depths, test_accs, label='Test')
    plt.xlabel("Max Tree-Depth")
    plt.ylabel("Präzision")
    plt.legend()
    plt.show()
    return train_accs, test_accs


def cross_validate_model(modelIN, features, targets, k, doFit=True, istKategTarget=True):
    """ Verwendet ein KFold, um ein Modell zu überprüfen, gibt zurück die Präzisionen """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True)
    test_accs = []
    model = modelIN
    for train_indices, test_indices in kf.split(features):
        train_features, test_features = features[train_indices], features[test_indices]
        train_targets, test_targets = targets[train_indices], targets[test_indices]
        if doFit:
            model = modelIN.fit(train_features, train_targets)
        test_accs.append(stat_op.median_diff(model.predict(test_features), test_targets,
                                             sindKlassen=istKategTarget))
    return test_accs


def cross_validate_predictions(modelIN, features, targets, k, doFit=False):
    """ Verwendet KFold um Voraussagen zu treffen """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True)
    all_predictions = np.zeros_like(targets)
    model = modelIN
    for train_indices, test_indices in kf.split(features):
        train_features, test_features = features[train_indices], features[test_indices]
        train_targets, test_targets = targets[train_indices], targets[test_indices]
        if doFit:
            model = modelIN.fit(train_features, train_targets)
        predictions = model.predict(test_features)
        all_predictions[test_indices] = predictions
    return all_predictions


def get_dt_img(dtIN, featureNames=None):
    """
        :param dt: Decision Tree
        :param featureNames: zB ['u - g', 'g - r', 'r - i', 'i - z']
        :return: ergibt Image in JPG Format
    """
    import pydotplus as pydotplus
    from sklearn.tree import export_graphviz
    if 'estimators_' in str(dir(dtIN)):
        print("Anzahl der Estimators: " + str(len(dtIN.estimators_)))
        xyN = int(input("Welcher estimator soll gezeigt werden: "))
        dt = dtIN.estimators_[xyN]
    else:
        dt = dtIN
    if featureNames is not None:
        dot_data = export_graphviz(dt, out_file=None, feature_names=featureNames)
    else:
        dot_data = export_graphviz(dt, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_jpg("./decision_tree.jpg")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
    plt.tight_layout()


def extract_metrics(clf, xdata, target, printout=False):
    _predicted = clf.predict(xdata[list(filter(lambda x: x != target, xdata.columns))].values)
    _targets = xdata[target].values
    TP = np.sum(list([x[0] == x[1] for x in list(zip(_predicted, _targets)) if x[1] == 1]))
    FP = np.sum(list([x[0] != x[1] for x in list(zip(_predicted, _targets)) if x[1] == 1]))
    FN = np.sum(list([x[0] != x[1] for x in list(zip(_predicted, _targets)) if x[1] == 0]))
    P1 = np.round(TP / (TP + FP), 2)
    R1 = np.round(TP / (TP + FN), 2)
    F1Score1 = np.round(2 * (P1 * R1) / (P1 + R1), 2)
    if printout:
        print(f"[x] TP: {TP}")
        print(f"[x] FP: {FP}")
        print(f"[x] FN: {FN}")
        print(f"=> P1:{P1}, R1:{R1}, F1-Score:{F1Score1}")
    return TP, FP, FN, P1, R1, F1Score1


def berechne_praezision(predicted, actual):
    nTrue = np.sum(list([x[0] == x[1] for x in list(zip(predicted, actual))]))
    return nTrue / predicted.shape[0]


def plot_dt_result(predicted, targets, class_labels=None):
    model_score = berechne_praezision(predicted, targets)
    print("[x] Präzision:", model_score)
    targetsPlot = targets
    predictedPlot = predicted
    if class_labels is None:
        class_labels = list(set(targets))
    else:
        targetsPlot = list([class_labels[x] for x in targets])
        predictedPlot = list([class_labels[x] for x in predicted])
    model_cm = confusion_matrix(y_true=targetsPlot, y_pred=predictedPlot, labels=class_labels)
    print(classification_report(predicted, targets, zero_division=0.0))
    plt.figure(figsize=(12, 8))
    plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
    plt.tight_layout()
    plt.show()
    return model_cm
