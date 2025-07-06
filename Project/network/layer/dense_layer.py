import numpy as np
import math

class DenseLayer:
    def __init__(self, input_size, output_size, Activation, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.Activation = Activation
        self.learning_rate = learning_rate
        self.weight_gradients = np.zeros((self.output_size, self.input_size))
        self.bias_gradients = np.zeros(output_size)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sums = np.dot(self.weights, inputs) + self.biases
        self.activations = self.Activation.func(self.weighted_sums)
        return self.activations

    def backward(self, activation_gradients):
        weighted_sum_gradients = activation_gradients * self.Activation.derivative(self.activations)
        self.weight_gradients += np.outer(weighted_sum_gradients, self.inputs)
        self.bias_gradients += weighted_sum_gradients
        input_gradients = np.dot(self.weights.T, weighted_sum_gradients)
        return input_gradients

    def update_weights(self, batch_size):
        self.weights -= self.weight_gradients * self.learning_rate / batch_size
        self.biases -= self.bias_gradients * self.learning_rate / batch_size

        self.weight_gradients = np.zeros(self.weight_gradients.shape)
        self.bias_gradients = np.zeros(self.bias_gradients.shape)

    def init_weights(self, init_method="random", distribution="uniform"):
        match init_method:
            case "zero":
                self.weights = np.zeros((self.output_size, self.input_size))
                self.biases = np.zeros(self.output_size)
            case "random":
                if distribution == "uniform":
                    self.weights = np.random.Generator.uniform(0, 1, (self.output_size, self.input_size))
                    self.biases = np.random.Generator.uniform(0,1, self.output_size)
                elif distribution == "normal":
                    self.weights = np.random.Generator.normal(0, 1, (self.output_size, self.input_size))
                    self.biases= np.random.Generator.normal(0, 1, self.output_size)
            case "xavier":
                if distribution == "uniform":
                    limit = math.sqrt(6/(self.input_size + self.output_size ))
                    self.weights = np.random.Generator.uniform(-limit, limit, (self.output_size, self.input_size))
                    self.biases = np.random.Generator.uniform(-limit, limit, self.output_size)
                elif distribution == "normal":
                    stdev = math.sqrt(6/(self.input_size + self.output_size))
                    self.weights = np.random.Generator.normal(0, stdev, (self.output_size, self.input_size))
                    self.biases = np.random.Generator.normal(0, stdev, self.output_size)
                else:
                    print("ERROR: Distribution method '" + distribution + "' not found for initialization method '" + self.init_method + "'")
            case "He":
                if distribution == "uniform":
                    min, max = -math.sqrt(6/self.input_size), math.sqrt(6/self.output_size)
                    self.weights = np.random.Generator.uniform(min, max, (self.output_size, self.input_size))
                    self.biases = np.random.Generator.uniform(min, max, self.output_size)
                elif distribution == "normal":
                    stdev = math.sqrt(2/self.input_size)
                    self.weights = np.random.Generator.normal(0, stdev, (self.output_size, self.input_size))
                    self.biases = np.random.Generator.normal(0, stdev, self.output_size)
                else:
                    print("ERROR: Distribution method '" + distribution + "' not found for initialization method '" + self.init_method + "'")