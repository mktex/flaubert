
from sklearn import metrics
from time import time
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

features_train = None
features_test = None
labels_train = None
labels_test = None
feature_names = None
target_name = None

tf_feature_columns = None

clf = None
clf_optim = None

tf.get_logger().setLevel('ERROR')

"""
Beispiel Verwendung:

from flaubert.model.neural_networks import tf_dnn
# feature_train, feature_test, labels_train, labels_test, feature_names und target_names
tf_dnn.set_data(ftr, fts, ltr, lts, fnames, tname)
tf_dnn.do_model(xtype='classification')
"""


def set_data(ftr, fts, ltr, lts, fnames, tname):
    global features_train, features_test, labels_train, labels_test, feature_names, target_name, tf_feature_columns
    features_train = np.array(ftr)
    features_test = np.array(fts)
    labels_train = np.array(ltr)
    labels_test = np.array(lts)
    feature_names = fnames
    target_name = tname
    tf_feature_columns = []
    for key in feature_names:
        tf_feature_columns.append(tf.feature_column.numeric_column(key=key))


def input_evaluation_set(features):
    global feature_names
    features_dict = {}
    for k in range(len(feature_names)):
        features_dict[feature_names[k]] = features[:, k]
    return features_dict


def input_fn(features, labels=None, training=True, batch_size=256):
    if labels is not None:  # Training
        dataset = tf.data.Dataset.from_tensor_slices((input_evaluation_set(features), labels))
    else:  # Voraussage
        dataset = tf.data.Dataset.from_tensor_slices(input_evaluation_set(features))
    if training and labels is not None:
        dataset = dataset.shuffle(batch_size).repeat()
    return dataset.batch(batch_size)


def do_model(xtype='classifier'):
    global clf, tf_feature_columns, features_train, labels_train
    if xtype == "classifier":
        clf = tf.estimator.DNNClassifier(
            feature_columns=tf_feature_columns,
            hidden_units=[30, 10],
            n_classes=len(set(labels_train)))
    elif xtype == "regression":
        clf = tf.estimator.DNNRegressor(
            feature_columns=tf_feature_columns,
            hidden_units=[1024, 512, 256],
            optimizer=lambda: tf.keras.optimizers.Adam(
                learning_rate=tf.compat.v1.train.exponential_decay(
                    learning_rate=0.1,
                    global_step=tf.compat.v1.train.get_global_step(),
                    decay_steps=10000,
                    decay_rate=0.96))
        )

    clf.train(input_fn=lambda: input_fn(features_train, labels_train, training=True), steps=5000)

    print_statistics(clf)


def do_model_optim():
    global clf, clf_optim
    print("Fitting the classifier to the training set")
    t0 = time()
    # min sample split maximum 30% of min of len labels_train and test
    xmax_samples = int(len(labels_train) * 0.3)
    xindex = [int(len(labels_train) * random.random()) for i in range(xmax_samples)]
    xindex.sort()
    xindex = list(set(xindex))
    lRate = (np.unique(np.random.randint(0, 200, 20)) / 1000.).tolist()
    param_grid = {
        'learning_rate': lRate,
        'dropout': [0.2, 0.8, None]
    }
    # for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
    # http://scikit-learn.org/stable/modules/model_evaluation.html
    clf_local = GridSearchCV(
        tf.estimator.DNNRegressor(
            hidden_units=[4, 4, 4],
            optimizer="Adam",
            dropout="dropout"
        ),
        param_grid,
        scoring='median_absolute_error'
    )
    clf_local = clf_local.fit(features_train[xindex, :], labels_train[xindex])
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf_local.best_estimator_)
    clf_optim = clf_local.best_estimator_
    # print_statistics(clf_optim)


def predict(new_data):
    global clf
    predictions = clf.predict(input_fn=lambda: input_fn(new_data))
    return list(predictions)


def print_statistics(xtype="classification"):
    """ Zeige Bewertung des Modells """
    global clf
    global features_test, labels_test, features_train, labels_train
    if xtype == "classification":
        eval_res_train = clf.evaluate(input_fn=lambda: input_fn(features_train, labels_train, training=False))
        eval_res_test = clf.evaluate(input_fn=lambda: input_fn(features_test, labels_test, training=False))
        pred_train = predict(features_train)
        pred_test = predict(features_test)
        labels_train_pred = [t["class_ids"][0] for t in pred_train]
        labels_test_pred = [t["class_ids"][0] for t in pred_test]
        print("Bewertung mit clf.evaluate():")
        print("TRAIN:", eval_res_train)
        print("TEST:", eval_res_test, "\n")
        print("Bewertung mit metrics.accuracy_score():")
        print('test: ', metrics.accuracy_score(labels_test, labels_test_pred))
        print('train:', metrics.accuracy_score(labels_train, labels_train_pred))
        print("TEST:")
        print('Classification Report')
        print(metrics.classification_report(labels_test, labels_test_pred))
        print('Confusion matrix')
        print(metrics.confusion_matrix(labels_test, labels_test_pred))
        print("")
        print("TRAIN:")
        print('Classification Report')
        print(metrics.classification_report(labels_train, labels_train_pred))
        print('Confusion matrix')
        print(metrics.confusion_matrix(labels_train, labels_train_pred))

    elif xtype == "regression":
        eval_res_train = clf.evaluate(input_fn=lambda: input_fn(features_train, labels_train, training=False))
        print(eval_res_train)
        pred_train = predict(features_train)
        target_train_pred = [t["predictions"][0] for t in pred_train]
        print('Train R2Sq: ', metrics.r2_score(target_train_pred, labels_train))
        print('Train MAE :', metrics.mean_absolute_error(target_train_pred, labels_train))
        plt.scatter(x=labels_train, y=target_train_pred, s=3)
        plt.xlabel("Train-Werte")
        plt.xlabel("Vprausgesagte Train-Werte")
        plt.show()
        eval_res_test = clf.evaluate(input_fn=lambda: input_fn(features_test, labels_test, training=False))
        print(eval_res_test)
        pred_test = predict(features_test)
        target_test_pred = [t["predictions"][0] for t in pred_test]
        print('Test R2Sq: ', metrics.r2_score(target_test_pred, labels_test))
        print('Test MAE :', metrics.mean_absolute_error(target_test_pred, labels_test))
        plt.xlabel("Test-Werte")
        plt.xlabel("Vprausgesagte Test-Werte")
        plt.show()
