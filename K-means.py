import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


########################################
#### DATA
MyData = False
if MyData:
     # Synthetic(Made Up) Data & Labels
    X_trg = np.array([[1,2],[3,4],[8,9],[10,11],[12,9],[13,13.5],[0,2],[1,6],[2,5]]) # Training data
    y_trg = np.array([0,0,1,1,1,1,0,0,0]) # Training labels (NOT REQUIRED by Kmeans!)
    
   

else:
    X_trg,y_trg = make_blobs(n_samples = 100, n_features = 2, centers = 3,cluster_std=1.5 ) #Makeing std smaller will make the blobs condensed

########################################
#### ALGORITHM

# Hyperparameters
K = 4 # number of centroids

# Training
model = KMeans(n_clusters = K,verbose = 0) # Pick an algorithm
model.fit(X_trg) # Do the ML/learning

yhat = model.labels_ # yhat is an estimate/guess of y_trg
print(model.cluster_centers_) # Locations of the Centroid computed by the algorithm
print(yhat) # Print answser

# Prediction
X_test = np.array([[8,4],[6,6],[3,9],[3,13]]) # New student
model.predict(X_test) # Labels predicted by Kmeans

########################################
# PLOTTING

# Plotting Clusters
c1 = model.cluster_centers_[:,0]
c2 = model.cluster_centers_[:,1]
plt.scatter(c1,c2)

# Pulling Out columns/features
height = X_trg[:,0] # Height
weight = X_trg[:,1] # Weight

# LEFT Plot
plt.subplot(1,2,1)
plt.scatter(height, weight, c=y_trg) # Colors are from y_trg
plt.grid()
plt.axis('equal')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Original Labels')

# RIGHT Plot
plt.subplot(1,2,2)
plt.scatter(height, weight, c=yhat) # Colors are from yhat

# Plotting Cluster Centers
x = 0
y = 0

for i in range(K):
    name = 'Cluster Center' + str(i+1)
    
    cords = model.cluster_centers_[x,y]
    y += 1
    cords2 = model.cluster_centers_[x,y]
    x +=1
    plt.text(cords,cords2,name)
    y = 0
    


#Final Touches
plt.grid()
plt.axis('equal')
plt.xlabel('Height')

plt.title('K-Means Labels (K = ' +str(K)+ ')')
