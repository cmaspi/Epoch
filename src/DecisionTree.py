import numpy as np
from DotDict import dotDict
from typing import Callable

class DecisionTree:
    """
    Decision Tree implementation in numpy
    """

    def __init__(self,
                criterion : str = 'gini',
                max_depth : int|None = None
                ) -> None:
        self.tree = dotDict({
            'feature' : None,
            'value' : None,
            'threshold' : None,
            'leftTree' : None,
            'rightTree' : None
        })
        self.criterion = criterion
        self.max_depth = max_depth
        self.used_features = set()

    @property
    def split_func(self):
        dispatch = {'gini' : self.giniIndex,
                    'entropy' : self.entropy}
        return dispatch[self.criterion]


    @staticmethod
    def giniIndex(p: float):
        """
        Gini Index of the given binary label data

        Args:
        -----
            p (float): the fraction of labels of 
            one particular value
        Returns:
            float: gini index_
        """
        return 2 * p * (1 - p)
    
    @staticmethod
    def entropy(p : float):
        """
        Entropy of the given binary label data

        Args:
        -----
            p (float): the fraction of labels of 
            one particular value
        Returns:
            float: Entropy
        """
        # to clamp the number in (0,1)
        p = sorted([1e-6, p, 1 - 1e-6])[1]
        return p * np.log2(p) + (1-p) * np.log2(1-p)
    
    @classmethod
    def infogain(cls,
                split_func : Callable[[float], float],
                y : np.ndarray,
                yl : np.ndarray,
                yr : np.ndarray
                ) -> float:
        """
        Finds the infogain using the given criterion (function),
        it expects the labels to be binary 0,1

        Args:
        -----
            split_func (Callable[[float], float]): criterion function
            y (np.ndarray): labels of entire data on that node
            yl (np.ndarray): labels separated to the left
            yr (np.ndarray): labels separated to the right

        Returns:
            float: _description_
        """
        n = y.size
        nl = yl.size
        nr = yr.size
        p = np.sum(y)/n
        pl = np.sum(yl)/nl
        pr = np.sum(yr)/nr
        return split_func(p) - (nl/n) * split_func(pl) - (nr/n) * split_func(pr)

    
    def bestSplit(self, 
                x : np.ndarray, 
                y : np.ndarray, 
                numThreshold : int = 10,
                ) -> dotDict:

        unique, counts = np.unique(y, return_counts = True)
        idx = max(enumerate(counts), key = lambda x: x[1])[0]
        value = unique[idx]

        # shape of input data
        n, m = x.shape

        BestinfoGain = -np.inf
        Bestthreshold = -1
        Bestfeature = -1
        Left, Right = (None,)*2

        for feature in range(m):
            if feature in self.used_features:
                continue
            allValsinFeature = x[:, feature]
            minVal = min(allValsinFeature)
            maxVal = max(allValsinFeature)
            Thresholds = np.linspace(minVal, maxVal, numThreshold+1, endpoint = False)[1:]

            for t in Thresholds:
                l = x[:, feature] < t
                lx = x[l]
                ly = y[l]
                r = l ^ True
                rx = x[r]
                ry = y[r]
                # p in l
                infogain = self.infogain(self.split_func, y,ly,ry)

                if infogain > BestinfoGain:
                    BestinfoGain = infogain
                    Bestfeature = feature
                    Bestthreshold = t
                    Left = lx,ly
                    Right = rx, ry


        return dotDict({
                'infogain' : BestinfoGain,
                'feature' : Bestfeature,
                'threshold' : Bestthreshold,
                'left' : Left,
                'right' : Right,
                'value' : value
            })


    def fit(self, X : np.ndarray, y : np.ndarray):
        """
        Fits the training data

        Args:
        -----
            X (np.ndarray): training data
            y (np.ndarray): labels
        """
        n,m = X.shape

        if self.max_depth is None:
            self.max_depth = m
        
        
        # Stopping Criteria
        if n > 1 and self.split_func(np.sum(y==y[0])/n) and self.used_features.__len__() < self.max_depth:
            best_split = self.bestSplit(X,y)
            self.tree.feature = best_split.feature
            self.tree.value = best_split.value
            self.tree.threshold = best_split.threshold
            self.used_features.add(self.tree.feature)
            self.tree.leftTree = DecisionTree()
            self.tree.leftTree.used_features = self.used_features.copy()
            self.tree.leftTree.fit(*best_split.left)
            self.tree.rightTree = DecisionTree()
            self.tree.rightTree.used_features = self.used_features.copy()
            self.tree.rightTree.fit(*best_split.right)
        else:
            unique, counts = np.unique(y, return_counts = True)
            idx = max(enumerate(counts), key = lambda x: x[1])[0]
            self.tree.value = unique[idx]
    
    def __classify__(self, x):
        if self.tree.feature is None:
            return self.tree.value
        if x[self.tree.feature] < self.tree.threshold:
            return self.tree.leftTree.__classify__(x)
        else:
            return self.tree.rightTree.__classify__(x)
    
    def predict(self, x):
        pred = np.zeros(x.shape[0])
        for i in range(pred.size):
            pred[i] = self.__classify__(x[i])
        return pred
            
            

        


            








