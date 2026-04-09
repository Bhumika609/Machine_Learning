import numpy as np
import matplotlib.pyplot as plt
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,1,1,0]
weights=np.array([10.0,0.2,-0.75])
alpha=0.05
def step(z):
    return 1 if z>=0 else 0
epochs=0
errors=[]
while epochs<1000:
    total_error=0
    for i in range(len(X)):
        x1,x2=X[i]
        target=Y[i]
        z=weights[0]+weights[1]*x1+weights[2]*x2
        output=step(z)
        error=target-output
        weights[0]+=alpha*error*1
        weights[1]+=alpha*error*x1
        weights[2]+=alpha*error*x2
        total_error+=error**2
    errors.append(total_error)
    epochs+=1
    if total_error<=0.002:
        break
print("Final weights",weights)
print("Epochs needed",epochs)
plt.plot(range(epochs),errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Epoch vs Error")
plt.grid()
plt.show()