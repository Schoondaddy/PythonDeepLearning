import numpy as np
from data_set import DataSet

train_data_path = 'MNIST_CSV/mnist_train.csv'
test_data_path = 'MNIST_CSV/mnist_test.csv'

class DataLoader:
    def parse_data_from_csv(self, path) -> DataSet:
        unparsed_data = np.loadtxt(path, delimiter=',')
        
        data_samples = []
        labels = []
        for i in unparsed_data:
            data_samples.append(i[1:])
            labels.append(i[0])
        
        return DataSet(np.array(data_samples), np.array(labels))