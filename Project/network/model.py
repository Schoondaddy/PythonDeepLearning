import numpy as np
from layer.dense_layer import DenseLayer
from data.data_set import DataSet

class Model:
    def __init__(self, layers, train : DataSet, test : DataSet, loss_function, learning_rate : float, mini_batch : bool = False, batch_size : int = 1):
        self.layers = layers
        self.train = train
        self.test = test
        self.loss_function = loss_function
        self.learning_rate = learning_rate
    
    def forward(data_set : DataSet):
        pass