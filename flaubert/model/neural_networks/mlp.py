import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from flaubert.statistik import stat_op

CLF_MLP = None
REG_MLP = None


def _mlp_train_test(Xin, Yin, ist_train_test_packet):
    if ist_train_test_packet:
        X_train, X_test = Xin
        y_train, y_test = Yin
    else:
        X, y = Xin.copy(), Yin.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)
    return X_train, X_test, y_train, y_test


def check_mlp(x, y):
    global CLF_MLP
    print("+++++++++++++++++++++++++++++++++++++")
    labels_zuordnung_mlp = CLF_MLP.classes_
    beispiel_mlp_x = x
    beispiel_mlp_y = y
    y_true = np.array(beispiel_mlp_y)
    y_pred = np.array([labels_zuordnung_mlp[np.argmax(t)] for t in CLF_MLP.predict_proba(beispiel_mlp_x)])
    accuracy = metrics.accuracy_score(y_true, y_pred)  # (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung_mlp)
    if True:
        print("Labels:", labels_zuordnung_mlp)
        print("Confusion Matrix:")
        print(cm)
        for i in range(0, len(cm)):
            precision, recall, f1_score = stat_op.get_confusion_matrix_stats(cm, i)
            print("Label {} - precision {}, recall {}, f1_score {}: ".format(
                i, np.round(precision, 2), np.round(recall, 2), np.round(f1_score, 2)
            ))
        print("accuracy:", accuracy)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=labels_zuordnung_mlp)
        disp.plot()
        plt.show()
    print("+++++++++++++++++++++++++++++++++++++")


def get_mlp_clf(Xin, Yin, ist_train_test_packet, lernrate=0.1, zyklen=100, struktur=(2, 2),
                batch_size=5, early_stopping=True, alpha=0.01,
                verbose=True):
    """
        Wenn ist_train_test_packet=True, dann Xin und Yin sind datenpackete mit:
            Xin = (Xin_train, Xin_test)
            Yin = (Yin_train, Yin_test)
    """
    global CLF_MLP
    X_train, X_test, y_train, y_test = _mlp_train_test(Xin, Yin, ist_train_test_packet)
    clf = MLPClassifier(hidden_layer_sizes=struktur, activation="logistic",
                        solver="adam", learning_rate="adaptive",
                        learning_rate_init=lernrate, max_iter=zyklen, batch_size=batch_size,
                        momentum=0.01, shuffle=True, verbose=verbose,
                        early_stopping=early_stopping, tol=1e-10,
                        n_iter_no_change=10000, alpha=alpha,
                        validation_fraction=0.3) \
        .fit(X_train, y_train)
    labels_zuordnung = clf.classes_
    print("Prediction on test: clf.predict_proba")
    print(pd.DataFrame(np.hstack([X_test, y_test.reshape(-1, 1)])).head(10))
    print("=>", clf.predict(X_test)[:10], "\n")
    print("\nLabels:", labels_zuordnung)
    y_true = np.array(y_test)
    y_pred = np.array([labels_zuordnung[np.argmax(t)] for t in clf.predict_proba(X_test)])
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(classification_report(y_pred, y_true, zero_division=0.0))
    cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nAccuracy (on Test):", accuracy)
    CLF_MLP = clf
    return clf


def get_mlp_reg(Xin, Yin, ist_train_test_packet,
                lernrate=0.8, zyklen=250, struktur=(2, 2,), batch_size=25, early_stopping=True, alpha=0.5,
                verbose=True):
    """
        Wenn ist_train_test_packet=True, dann Xin und Yin sind datenpackete mit:
            Xin = (Xin_train, Xin_test)
            Yin = (Yin_train, Yin_test)
    """
    global REG_MLP
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error
    X_train, X_test, y_train, y_test = _mlp_train_test(Xin, Yin, ist_train_test_packet)
    reg = MLPRegressor(hidden_layer_sizes=struktur, activation="logistic",
                       solver="adam", learning_rate="adaptive",
                       learning_rate_init=lernrate, max_iter=zyklen, batch_size=batch_size,
                       momentum=0.0, shuffle=True, verbose=verbose, alpha=alpha,
                       validation_fraction=0.2,
                       early_stopping=early_stopping, tol=1e-10, n_iter_no_change=10000)
    reg.fit(X_train, y_train)
    REG_MLP = reg
    xNNOutputPred = reg.predict(X_test)
    print("[x] RMSE:", mean_squared_error(xNNOutputPred.flatten(), y_test.flatten()))
    return reg
