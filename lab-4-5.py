import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
X_train=np.random.uniform(1,10,(20,2))
Y_train=np.random.randint(0,2,20)#this is to assign the class labels
neigh=KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train,Y_train)
X_test=np.random.uniform(0,10,(10000,2))
y_test_pred=neigh.predict(X_test)
#plotting the test points
plt.figure()
plt.scatter(X_test[:,0],X_test[:,1],c=y_test_pred,alpha=0.2)#[:,0]represents the X and [:,1]represents teh y coordinate
# Plot training points (bold colors)
plt.scatter(X_train[:,0],X_train[:,1],c=Y_train,edgecolor='black')
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("KNN Decision Regions")
plt.show()
#purple area is the model predicts class0
#yellow are teh model predicts class1
#the original datapoints are bold in color which has edge color.
#as the k value increases the decision is very influenced by the majority overall which leads to overfiitng
#if my k value is intialized to 1 teh value is less means decreasing then there is no votnig to classifying the class is directly given based on the neighbor it gives an extremely irregular boundary.all neraby test ponits will be classified wrong which leads to overfitting.