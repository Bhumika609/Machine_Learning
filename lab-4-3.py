import numpy as np
import matplotlib.pyplot as plt
X=np.random.uniform(0,10,20)
Y=np.random.uniform(0,10,20)
classes=np.random.randint(0,2,20)
print(X)
print(Y)
for i in range(20):
    if classes[i]==0:
        plt.scatter(X[i],Y[i],color='blue')
    else:
        plt.scatter(X[i],Y[i],color='red')
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.show()