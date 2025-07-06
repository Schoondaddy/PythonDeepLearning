import numpy as np

class DataSet:
    def __init__(self, data_samples: np.ndarray, labels: np.ndarray):
        assert data_samples.shape[0] == labels.shape[0], "ERROR: Dataset size must be equal to label size"
        self.data_samples = data_samples
        self.labels = labels
    
    def shuffle(self):
        indices = np.random.permutation(self.data_samples.shape[0])
        self.data_samples = self.data_samples[indices]
        self.labels = self.labels[indices]
    
    def get_data_sample(self, index: int) -> np.ndarray:
        return self.data_samples[index]
    
    def get_data_set(self) -> np.ndarray:
        return self.data_samples