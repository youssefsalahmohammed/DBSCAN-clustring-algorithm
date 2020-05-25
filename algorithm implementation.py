import numpy as np  # importing used and required libraries
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


centers = [[1, 1], [-1, -1], [1, -1]] # our base of the data set
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0) # make_blobs() functions generate isotropic Gaussian blobs for clustering
X = StandardScaler().fit_transform(X)


db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True



n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1) 



unique_labels = set(labels) # in order to convert the lables array into set {}

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]  # here we create colors array using numpy library ( with the length of unique_labels) to be merged with
                                                              # the unique_labels set in the next loop.

# zip function used to merge or join two sets and arrays together
for k, col in zip(unique_labels, colors): 
    if k == -1: # -1 means it is a noise 
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14) # plot function are using to draw the points of clusters 

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)) # plot function are using to draw the points of clusters

plt.title('Estimated number of clusters: %d' % n_clusters_) # setting title of the plot that will be shown next
plt.show() #function that will output the plot result 
