import numpy as np
from scipy import stats
from copy import deepcopy
from typing import Callable
from sklearn.neighbors import KNeighborsClassifier


clf = KNeighborsClassifier
class KNN:
    """
    KNN: K-Nearest Neighbours
    """
    def __init__(self, 
                n_neighbors: int = 3,
                weights: str | Callable = 'uniform',
                algorithm: str = 'auto',
                distance: str | Callable = 'minkowski',
                p : int = 2
                ) -> None:
        """
        K Nearest Neighbour classifer

        Args:
        -----
            - n_neighbors (int, optional): Number of neighbours. Defaults to 3.

            - weights (str | Callable, optional): Weight function. Defaults to 'uniform'.   
                Possible Values: 
                    - uniform : same weight
                    - weighted : Weighted by inverse of the distance
                    - [Callable] : A function that takes in array of distances
                        and returns an array of weights of the same shape

            - algorithm (str, optional): Algorithm to use. Defaults to 'auto'.  
                Possible Values:
                    - brute : brute forces the solution
                    - kd_tree : uses the k-d tree algorithm
                    - ball_tree : uses the ball tree algorithm
                    - auto : finds the best algorithm to use by itself
                
            - distance (str | Callable, optional): distance function between two points. 
                    Defaults to 'minkowski'.
                Possible Values:
                    - minkowski : p-norm
                    - manhattan : p-norm with p = 1
                    - euclidian : p-norm with p = 2
                    - [Callable] : function that takes two datapoints and returns the distance
                        between them
            
            - p (int, optional): value of p if distance is minkowski
        """
        self.n_neighbours = n_neighbors

        # checking if distance parameter is callable
        # if callable, assign the function to instance attribute
        if callable(weights):
            self.weights = weights
        else:
            if weights == 'uniform':
                self.weights = lambda x : np.ones_like(x)
            elif weights == 'weighted':
                self.weights = lambda distance : 1/(1e-9 + distance) # ensuring distance > 0
            else:
                raise ValueError(f"{weights} is not a valid parameter for the weights argument")
        
        if algorithm not in ["brute", "kd_tree", "ball_tree", "auto"]:
            raise ValueError(f"{algorithm} is not a valid parameter for the algorithm argument")
        self.algorithm = algorithm

        # checking if distance parameter is callable
        # if callable, assign the function to instance attribute
        if callable(distance):
            self.distance = distance
        else:
            if distance == 'manhattan':
                self.distance = lambda x1, x2 : np.linalg.norm(x1-x2, 1)
            elif distance == 'euclidian':
                self.distance = lambda x1, x2 : np.linalg.norm(x1-x2, 2)
            elif distance == 'minkowski':
                self.distance = lambda x1, x2 : np.linalg.norm(x1-x2, p)
            else:
                raise ValueError(f"{distance} is not a valid parameter for the distance argument")
        

        
