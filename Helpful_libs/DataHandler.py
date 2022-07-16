from typing import Iterable, Tuple
import numpy as np


def shuffle(train: np.ndarray,
            labels: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffles the data in a random permutation
    """
    n = train.shape[0]
    permutation = np.random.permutation(n)
    train = train[permutation]
    labels = labels[permutation]
    return train, labels


def iterate_minibatches(train: np.ndarray,
                        labels: np.ndarray,
                        batch_size: int
                        ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Gives a generator object of mini-batches
    """
    train, labels = shuffle(train, labels)
    for start_idx in range(0, train.shape[0] + 1 - batch_size, batch_size):
        excerpt = slice(start_idx, start_idx+batch_size)
        yield train[excerpt], labels[excerpt]


def split(x: np.ndarray,
          y: np.ndarray,
          train: float = 0.7,
          validation: float = 0.15,
          shuffle_bool: bool = True
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                     np.ndarray, np.ndarray]:
    """
    Returns a train - validation - test split

    Args:
    -----
        x (np.ndarray): data
        y (np.ndarray): labels
        train (float, optional): fraction of training samples. Defaults to 0.7.
        validation (float, optional): fraction of validation samples. Defaults
                                      to 0.15.
        test (float, optional): fraction of test samples. Defaults to 0.15.

    Returns:
    --------
        Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray,
              np.ndarray]
    """
    n = y.size
    a = int(train*n)
    b = int((train+validation)*n)
    if shuffle_bool:
        x, y = shuffle(x, y)
    return x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:]
