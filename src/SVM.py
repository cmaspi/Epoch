import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

find_y = lambda x,w,b : (-b-w[0]*x)/w[1]

class svm:
    def __init__(self, trade_off: int = 10):
        self.w = None
        self.b = None
        self.trade_off = trade_off
        self.learning_rate = None
        self.best_w = None
        self.best_b = None
        self.best_loss = 1e9
        self.prev_grad_w = None
        self.prev_grab_b = 0
        

    @staticmethod
    def shuffle(train: np.ndarray, labels: np.ndarray):
        """
        Shuffles the data in a random permutation
        """
        n = train.shape[0]
        permutation = np.random.permutation(n)
        train = train[permutation]
        labels = labels[permutation]
        return train, labels
    
    def iterate_minibatches(self, train : np.ndarray, labels: np.ndarray, batch_size : int):
        """
        Gives minibatches
        """
        train, labels = self.shuffle(train,labels)
        for start_idx in range(0, train.shape[0] + 1 - batch_size, batch_size):
            excerpt = slice(start_idx, start_idx+batch_size)
            yield train[excerpt], labels[excerpt]

    def loss(self, train : np.ndarray, labels : np.ndarray):
        """
        Returns the loss
        """
        return self.w @ self.w + self.trade_off * np.mean(np.maximum( 0 , 1 - labels * (train @ self.w + self.b)))

    
    def fit_util(self, train: np.ndarray, labels: np.ndarray, momentum : int):
        """
        Utility function for fit
        """
        indices = (labels * (train @ self.w + self.b )) < 1.
        grad_b = -self.trade_off * np.sum(labels[indices]) / (train.shape[0] if train.shape[0] else 1)
        grad_w = 2 * self.w  - self.trade_off * (labels[indices] @ train[indices]) / (train.shape[0] if train.shape[0] else 1)
        self.b -=  self.learning_rate * grad_b + momentum * self.prev_grab_b
        self.w -= self.learning_rate * grad_w + momentum * self.prev_grad_w
        self.prev_grab_b = self.learning_rate * grad_b + momentum * self.prev_grab_b
        self.prev_grad_w = self.learning_rate * grad_w + momentum * self.prev_grad_w
        
        
    def fit(self, train : np.ndarray, labels : np.ndarray, learning_rate : float = 0.1 ,epochs : int = 500, batch_size = 4, fig: bool = False, momentum : int = 0):
        """
        Trains the weights
        """
        self.learning_rate = learning_rate
        s = train.shape[1]
        self.w = np.random.normal(np.zeros(s, dtype = float))
        self.b = 0
        self.prev_grad_w = np.zeros(self.w.shape)
        
        for epoch in range(epochs):
            train, labels = self.shuffle(train, labels)
            for x,y in self.iterate_minibatches(train, labels, batch_size):
                self.fit_util(x, y, momentum)
                
            if self.best_loss > self.loss(train, labels):
                self.best_b = self.b
                self.best_w = self.w.copy()
                self.best_loss = self.loss(train, labels)
            
            if fig:
                fig, ax = plt.subplots()
                ax.set_ylim(-60,400)
                ax.set_xlim(-50,250)
                ax.scatter( train[ labels == 1, 0], train[labels == 1, 1], label = 1)
                ax.scatter( train[ labels == -1, 0], train[labels == -1, 1], label = -1)

                x_coords = np.array([-50.0, 250.0])
                dist = 1/self.w[1]
                ax.plot( x_coords, find_y(x_coords, self.w, self.b), label = "separating_line" )

                ax.fill_between(x_coords, find_y(x_coords, self.w, self.b) - dist, find_y(x_coords, self.w, self.b)+dist, label = "slab", edgecolor = None, alpha=0.4, color='white')

                ax.legend(loc='upper right')
                plt.savefig(f'fig/{epoch}.png')
                plt.close()
                clear_output()
        self.w = self.best_w.copy()
        self.b = self.best_b
            

    def display(self, train : np.ndarray, labels: np.ndarray, dpi : int = 500):
        fig, ax = plt.subplots(dpi = dpi)
        ax.set_xlim(-0.5,1.5)
        ax.set_ylim(-0.5,1.5)
        ax.scatter( train[ labels == 1, 0], train[labels == 1, 1], label = 1)
        ax.scatter( train[ labels == -1, 0], train[labels == -1, 1], label = -1)

        x_coords = np.array([-0.6, 1.6])
        dist = 1/self.w[1]
        ax.plot( x_coords, find_y(x_coords, self.w, self.b), label = "separating_line" )

        ax.fill_between(x_coords, find_y(x_coords, self.w, self.b) - dist, find_y(x_coords, self.w, self.b)+dist, label = "slab", edgecolor = None, alpha=0.4, color='white')

        ax.legend(loc='upper right')
        return ax

    def predict(self, test : np.ndarray):
        """
        returns the predicted label
        """
        logits = test @ self.w + self.b
        return np.sign(logits)
    
    def accuracy(self, test : np.ndarray, ground_truth : np.ndarray):
        """
        Returns the accuracy of the given model
        test datapoints, and ground truth for those 
        datapoints"""
        predicted = self.predict(test)
        return np.mean(predicted == ground_truth)

