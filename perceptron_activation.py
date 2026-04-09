import numpy as np
import matplotlib.pyplot as plt
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,0,0,1]
Y_bipolar=[-1,-1,-1,-1]
weights_relu=np.array([10.0,0.2,-0.75])
weights_bipolar=np.array([10.0,0.2,-0.75])
weights_sigmoid=np.array([10.0,0.2,-0.75])
alpha=0.05
def relu(z):
    return max(0,z)
def bipolar_step(z):
    return 1 if z>=0 else -1
def sigmoid(z):
    return 1/(1+np.exp(-z))
epochs=0
errors_sigmoid=[]
errors_relu=[]
errors_bipolar=[]
while epochs<1000:
    total_error_sigmoid=0
    total_error_relu=0
    total_error_bipolar=0
    for i in range(len(X)):
        x1,x2=X[i]
        target=Y[i]
        z=weights_sigmoid[0]+weights_sigmoid[1]*x1+weights_sigmoid[2]*x2
        output_sigmoid=sigmoid(z)
        error_sigmoid=target-output_sigmoid
        weights_sigmoid[0]+=alpha*error_sigmoid*1
        weights_sigmoid[1]+=alpha*error_sigmoid*x1
        weights_sigmoid[2]+=alpha*error_sigmoid*x2
        total_error_sigmoid+=error_sigmoid**2


        z=weights_relu[0]+weights_relu[1]*x1+weights_relu[2]*x2
        output_relu=relu(z)
        error_relu=target-output_relu
        weights_relu[0]+=alpha*error_relu*1
        weights_relu[1]+=alpha*error_relu*x1
        weights_relu[2]+=alpha*error_relu*x2
        total_error_relu+=error_relu**2


        z=weights_bipolar[0]+weights_bipolar[1]*x1+weights_bipolar[2]*x2
        output_bipolar=bipolar_step(z)
        error_bipolar=Y_bipolar[i]-output_bipolar
        weights_bipolar[0]+=alpha*error_bipolar*1
        weights_bipolar[1]+=alpha*error_bipolar*x1
        weights_bipolar[2]+=alpha*error_bipolar*x2
        total_error_bipolar+=error_bipolar**2
    errors_relu.append(total_error_relu)
    errors_sigmoid.append(total_error_sigmoid)
    errors_bipolar.append(total_error_bipolar)
    epochs+=1
    if (total_error_sigmoid <= 0.002 and 
        total_error_relu <= 0.002 and 
        total_error_bipolar <= 0.002):
        break
print("Relu weights",weights_relu)
print("Sigmoid weughts",weights_sigmoid)
print("Bipolar weights",weights_bipolar)
print("Epochs needed",epochs)
plt.plot(errors_sigmoid, label="Sigmoid")
plt.plot(errors_relu, label="ReLU")
plt.plot(errors_bipolar, label="Bipolar")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Activation Comparison")
plt.legend()
plt.grid()
plt.show()