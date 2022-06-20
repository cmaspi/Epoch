import numpy as np

def shuffle(train: np.ndarray, labels: np.ndarray):
    """
    Shuffles the data in a random permutation
    """
    n = train.shape[0]
    permutation = np.random.permutation(n)
    train = train[permutation]
    labels = labels[permutation]
    return train, labels

def iterate_minibatches(train : np.ndarray, labels: np.ndarray, batch_size : int):
    """
    Gives minibatches
    """
    train, labels = shuffle(train,labels)
    for start_idx in range(0, train.shape[0] + 1 - batch_size, batch_size):
        excerpt = slice(start_idx, start_idx+batch_size)
        yield train[excerpt], labels[excerpt]