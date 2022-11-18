# VERY SIMPLE
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1,2],[3,4],[8,9],[10,11]])
model = KMeans(n_clusters = 2)
model.fit(X) #do the ML
print(model.cluster_centers_)
print(model.labels_)
