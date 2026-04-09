import numpy as np
X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[0],[0],[1]])
V=np.random.uniform(-1,1,(2,2))
W=np.random.uniform(-1,1,(2,1))
alpha=0.05
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(x):
    return x*(1-x)
epochs=0
while epochs<1000:
    H_in=np.dot(X,V)
    H_out=sigmoid(H_in)
    O_in=np.dot(H_out,W)
    O_out=sigmoid(O_in)
    error=Y-O_out
    total_error=np.sum(error**2)
    #back propogation
    do=error*sigmoid_derivative(O_out)
    dh=do.dot(W.T)*sigmoid_derivative(H_out)
    W+=alpha*H_out.T.dot(do)
    V+=alpha*X.T.dot(dh)
    epochs+=1
    if total_error<0.002:
        break
print("Epochs",epochs)
print("Output",O_out.round())