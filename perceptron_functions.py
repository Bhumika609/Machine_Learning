import numpy as np
def summation(x,weights):
    return weights[0]+weights[1]*x[0]+weights[2]*x[1]
def step(z):
    return 1 if z>=0 else 0
def bipolar_step(z):
    return 1 if z>=0 else -1
def sigmoid(z):
    return 1/(1+np.exp(-z))
def tanh(z):
    return np.tanh(z)
def relu(z):
    return max(0,z)
def leaky_relu(z):
    return z if z>0 else 0.01*z
def compute_error(target,output):
    return target-output