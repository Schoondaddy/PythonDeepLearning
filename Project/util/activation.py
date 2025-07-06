import numpy as np
import math

class Activation:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative
np.fft
## Sigmoid
def sigmoid(input: list[float]) -> list[float]:
    output = float[len(input)]
    for i in range(len(output)):
        output[i] = 1/(1 + math.exp(-input[i]))
    return output
def sigmoid_derivative(input: list[float]) -> list[float]:
    output = float[len(input)]
    for i in range(len(output)):
        output[i] = sigmoid(input[i])*(1 - sigmoid(input[i]))
    return output

## Rectified Linear Unit
def ReLU(input: list[float]) -> list[float]:
    output = float[len(input)]
    for i in range(len(output)):
        output[i] = 0.0 if input[i] < 0 else input[i]
    return output
def ReLU_derivative(input: list[float]) -> list[float]:
    output = float[len(input)]
    for i in range(len(output)):
        output[i] = 0.0 if input[i] < 0 else 1
    return output

## Leaky Rectified Linear Unit
def leaky_ReLU(input: list[float]) -> list[float]:
    output = float[len(input)]
    for i in range(len(output)):
        output[i] = 0.01 * input[i] if input[i] < 0 else input[i]
    return output
def leaky_ReLU_derivative(input: list[float]) -> list[float]:
    output = float[len(input)]
    for i in range(len(output)):
        output[i] = 0.01 if input[i] < 0 else 1
    return output
