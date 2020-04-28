import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(X.T.dot(X),X.T.dot(y))
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        new_X = np.zeros((X.shape[0], k+1))
        for d in range(k+1):
            new_X[:, d] = (X ** d).reshape((X.shape[0]))
        return new_X
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        new_X = np.ones((X.shape[0], k+2))
        new_X[:, :(k+1)] = self.create_poly(k, X)
        new_X[:,(k+1)] = np.sin(new_X[:,1]).reshape((new_X.shape[0]))
        return new_X
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return X.dot(self.theta)
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
    x_train,y_train=util.load_dataset(train_path)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()

    for i in range(len(ks)):
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel()

        if(sine):
            new_X_train = model.create_sin(ks[i], x_train)
        else:
            new_X_train = model.create_poly(ks[i], x_train)

        model.fit(new_X_train, y_train)

        if(sine):
            plot_y = model.predict(model.create_sin(ks[i], plot_x[:, 1]))
        else:
            plot_y = model.predict(model.create_poly(ks[i], plot_x[:, 1]))

        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.scatter(x_train, y_train, color='black')
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % ks[i])

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, False, ks=[3], filename='4b.png', )
    run_exp(train_path, False, ks=[3,5,10,20], filename='4c.png')
    run_exp(train_path, True, ks=[0,1,2,3,5,10,20], filename='4d.png')
    run_exp(small_path, False, ks=[1,2,5,10,20], filename='4e.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
