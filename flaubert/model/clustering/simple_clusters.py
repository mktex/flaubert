import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
from sklearn import cluster

figsize = (12, 8)
point_size = 150
point_border = 0.8


def plot_clustered_dataset(dataset, y_pred, xlim=(-15, 15), ylim=(-15, 15), neighborhood=False, epsilon=0.5):
    """ Quelle: Udacity """
    fig, ax = plt.subplots(figsize=figsize)
    colors = np.array(list(islice(cycle(['#df8efd', '#78c465', '#ff8e34',
                                         '#f65e97', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_pred) + 1))))
    colors = np.append(colors, '#BECBD6')
    if neighborhood:
        for point in dataset:
            circle1 = plt.Circle(point, epsilon, color='#666666', fill=False, zorder=0, alpha=0.3)
            ax.add_artist(circle1)
    ax.scatter(dataset[:, 0], dataset[:, 1], s=point_size, color=colors[y_pred], zorder=10, edgecolor='black',
               lw=point_border)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_dbscan_grid(dataset, eps_values, min_samples_values):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=.02, right=.98, bottom=0.001, top=.96, wspace=.05,
                        hspace=0.25)
    plot_num = 1
    for i, min_samples in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            ax = fig.add_subplot(len(min_samples_values), len(eps_values), plot_num)
            dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
            y_pred_2 = dbscan.fit_predict(dataset)
            colors = np.array(list(islice(cycle(['#df8efd', '#78c465', '#ff8e34',
                                                 '#f65e97', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred_2) + 1))))
            colors = np.append(colors, '#BECBD6')
            for point in dataset:
                circle1 = plt.Circle(point, eps, color='#666666', fill=False, zorder=0, alpha=0.3)
                ax.add_artist(circle1)

            ax.text(0, -0.03, 'Epsilon: {} \nMin_samples: {}'.format(eps, min_samples), transform=ax.transAxes,
                    fontsize=12, va='top')
            ax.scatter(dataset[:, 0], dataset[:, 1], s=50, color=colors[y_pred_2], zorder=10, edgecolor='black', lw=0.5)
            plt.xticks(())
            plt.yticks(())
            plt.xlim(-14, 5)
            plt.ylim(-12, 7)
            plot_num = plot_num + 1
    plt.tight_layout()
    plt.show()


def plot_dataset(dataset, xlim=(-15, 15), ylim=(-15, 15)):
    plt.figure(figsize=figsize)
    plt.scatter(dataset[:, 0], dataset[:, 1], s=point_size, color="#00B3E9", edgecolor='black', lw=point_border)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def check_clusters(_data, _center):
    kmeans = KMeans(n_clusters=_center)
    model = kmeans.fit(_data)
    score = np.abs(model.score(_data))
    plt.plot(list(range(15)), scores, linestyle='--', marker='o', color='b')
    return score


data, labels = make_blobs(n_samples=100, n_features=2, centers=4)

plot_dataset(data)

# DBSCAN Cluster
epsilon= 1.5
dbscan = cluster.DBSCAN(eps=epsilon, min_samples=5)
cl_labels = dbscan.fit_predict(data)
plot_clustered_dataset(data, cl_labels, neighborhood=True, epsilon=epsilon)

eps_values = [0.3, 0.5, 1, 1.3, 1.5]
min_samples_values = [2, 5, 10, 20, 80]
plot_dbscan_grid(data, eps_values, min_samples_values)

scores = []
for center in range(15):
    scores.append(check_clusters(data, center))

