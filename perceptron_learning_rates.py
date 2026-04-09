import numpy as np
import matplotlib.pyplot as plt
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,0,0,1]
weights=np.array([10.0,0.2,-0.75])
alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
def step(z):
    return 1 if z>=0 else 0
epochs=0
errors=[]
epochs_list=[]
for a in alpha:
    weights=np.array([10.0,0.2,-0.75])
    epochs=0
    while epochs<1000:
        total_error=0
        for i in range(len(X)):
            x1,x2=X[i]
            target=Y[i]
            z=weights[0]+weights[1]*x1+weights[2]*x2
            output=step(z)
            error=target-output
            weights[0]+=a*error*1
            weights[1]+=a*error*x1
            weights[2]+=a*error*x2
            total_error+=error**2
        epochs+=1
        if total_error<=0.002:
            break
    epochs_list.append(epochs)
for i in range(len(alpha)):
    print(f"LR={alpha[i]} → Epochs={epochs_list[i]}")

# Plot
plt.plot(alpha, epochs_list, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("Learning Rate vs Epochs")
plt.grid()
plt.show()