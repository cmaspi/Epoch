import numpy as np
from scipy import stats
from copy import deepcopy

class KNN:
    def __init__(self, n_neighbours : int = 5) -> None:
        self.n_neighbours = n_neighbours
        self.x = None
        self.y = None
    
    @staticmethod
    def dist(x1, x2) -> float:
        """
        Returns the euclidian distance

        Args:
        -----
            x1 (np.ndarray): 1st point
            x2 (np.ndarray): 2nd point

        Returns:
            float: euclidian distance
        """
        return np.linalg.norm(x1-x2, axis = 1)
    
    def fit(self, x : np.ndarray, y : np.ndarray) -> None:
        """
        just saves the training data

        Args:
        -----
            x (np.ndarray): training data
            y (np.ndarray): training labels
        """
        self.x = deepcopy(x)
        self.y = deepcopy(y)
    
    def __get_label__(self, x):
        """
        Helper function, returns the predicted label for a 
        given test data point

        Args:
        -----
            x (np.ndrray): data point
        
        Returns
        -------
            any: predicted label
        """
        distances = self.dist(x, self.x)
        neighbours = np.argsort(distances)[:self.n_neighbours]
        label = stats.mode(self.y[neighbours])[0][0]
        return label
    
    def predict(self, test_x : np.ndarray) -> np.ndarray:
        """
        predicts the labels of given test data input

        Args:
            test_x (np.ndarray): data

        Returns:
            np.ndarray: labels
        """
        n = test_x.shape[0]
        labels = np.zeros(n)
        for i in range(n):
            labels[i] = self.__get_label__(test_x[i])
        return labels


        