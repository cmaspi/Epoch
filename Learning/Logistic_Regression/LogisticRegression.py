import sys
sys.path.insert(0, "../../")
import numpy as np
from tqdm import tqdm
from Helpful_libs.DataHandler import shuffle, iterate_minibatches


class logisticRegression:
    """
    (Binary) Logistic Regression is a statistical model that
    models probability of an event by taking the log of odds.
    """
    W = np.zeros(1)
    b = 0

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid function

        Args:
        -----
            x (np.ndarray): n x 1 vector

        Returns:
        --------
            np.ndarray: n x 1 vector
        """
        return 1/(1+np.exp(-x))

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 10,
            batch_size: int = 32,
            lr: float = 1e-4
            ):
        """
        learns the mapping from data to label

        Args:
        ----
            x (np.ndarray): training data
            y (np.ndarray): training labels
            epochs (int, optional): Number of epochs. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 32.
            lr (float, optional): learning rate. Defaults to 1e-4.
        """
        # shuffles the data
        x, y = shuffle(x, y)

        # initializing weights
        self.W = np.zeros(x.shape[1])
        self.b = 0

        # using tqdm for better progress bar
        for _ in tqdm(range(epochs)):
            # iterating in mini batches
            for train, label in iterate_minibatches(x, y, batch_size):
                delta_b = lr * (self.__logits__(train) - label)
                self.W -= train.T @ delta_b / batch_size
                self.b -= np.mean(delta_b)

    def __logits__(self, x: np.ndarray) -> np.ndarray:
        """
        returns the logits

        Args:
        -----
            x (np.ndarray): data

        Returns:
        --------
            np.ndarray: logits
        """
        return self.sigmoid(x @ self.W + self.b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        rounds of the logits

        Args:
        -----
            x (np.ndarray): data

        Returns:
        --------
            np.ndarray: label
        """
        return np.around(self.__logits__(x))
