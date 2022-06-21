from DecisionTree import DecisionTreeClassifier
import numpy as np
from DotDict import dotDict
from DataHandler import shuffle
from scipy.stats import mode

class RandomForestClassifier:
    """
    Random forest is an ensemble method
    that is based on the principle of bagging
    """

    def __init__(self, 
                n_trees : int = 100, 
                criterion : str = 'gini',
                n_thresholds : int = 10
                ) -> None:
        
        self.forest = [dotDict({'clf' : DecisionTreeClassifier(
            criterion = criterion,
            n_thresholds = n_thresholds 
        ),
        'features' : None}) for _ in range(n_trees)]

        self.n_trees = n_trees
    
    def fit(self, train_x : np.ndarray, train_y : np.ndarray) -> None:
        n, m = train_x.shape
        for tree in self.forest:
            data, labels = shuffle(train_x, train_y)
            data, labels = data[:n//3], labels[:n//3]
            f = np.random.choice(m, 3*m//5)
            data = data[:,f]
            tree.features = f
            tree.clf.fit(data, labels)
    
    def predict(self, x):
        results, pred = np.zeros((self.n_trees, x.shape[0])), np.zeros(x.shape[0])
        for i,tree in enumerate(self.forest):
            f = tree.features
            results[i] = tree.clf.predict(x[:,f])
            
        for i in range(x.shape[0]):
            pred[i] = mode(results[:,i])[0][0]
        return pred



    
    


        