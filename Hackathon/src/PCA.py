import scipy.linalg as sc
import numpy as np

class PCA:
    """
    PCA is a dimensionality reduction method
    """
    def __init__(self, n_components : int) -> None:
        self.n_components = n_components
        self.means = None
        self.frac_variance = None
        self.covered_variance = None
        self.evecs = None
    
    @property
    def explain_variance(self):
        print(f'Variance Covered : {self.covered_variance}\nFraction Variance : {self.frac_variance}')


    def __normalize__(self, data : np.ndarray) -> np.ndarray:
        """
        normalizes the data, if __fit_normalize__ isn't called before
        this method then it would run that instead.

        Args:
            data (np.ndarray): the input data

        Returns:
            np.ndarray: data with reduced dimensionality
        """
        if self.means is None: # if __fit_normalize wasn't called before
            return self.__fit_normalize__(data)

        # normalizing the data
        data -= self.means
        data /= self.std
        return data

    def __fit_normalize__(self, data : np.ndarray) -> np.ndarray:
        """
        normalizes the data and saves the mean and standard deviation 
        for reuse by __normalize__

        Args:
            data (np.ndarray): the input data

        Returns:
            np.ndarray: data with reduced dimensionality
        """
        self.means = np.mean(data, axis = 0)
        data -= self.means # making the mean zero
        self.std = data.std(axis = 0)
        data /= self.std
        return data
    
    def transform(self, data : np.ndarray) -> np.ndarray:
        """
        Transforms the given data to limit it to k dimensions

        Args:
            data (np.ndarray): input data

        Returns:
            np.ndarray: dimensionality reduced data
        """
        if self.means is None:
            return self.fit_transform()
        return data @ self.evecs 

    def fit_transform(self, data : np.ndarray) -> np.ndarray:
        """
        fits and transforms the given data to limit it to k dimensions

        Args:
            data (np.ndarray): input data

        Returns:
            np.ndarray: dimensionality reduced data
        """
        data = self.__fit_normalize__(data)
        cov = np.cov(data, rowvar = False) 
        evals, evecs = sc.eigh(cov)
        # scipy evals are sorted in ascending order by default
        # reversing them (descending order)
        evals, evecs = evals[::-1], evecs[:,::-1]
        # selecting first n_components
        self.evecs = evecs[:,:self.n_components]
        # sums of eigen values to find fraction variance ration
        total = evals.sum()
        self.frac_variance = evals[:self.n_components]/total
        self.covered_variance = evals[:self.n_components].sum()/total
        return data @ self.evecs 