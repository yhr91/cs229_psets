import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)

    # Train a logistic regression classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_eval.T)

    # Plot decision boundary on top of validation set set
    theta = np.concatenate([[clf.theta0], clf.theta])
    util.plot(x_eval,y_eval,theta,save_path.split('.')[0]+'.jpeg')

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path,predictions)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=np.zeros([2,1]), verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.theta0 = 0

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        ind_0 = np.where(y == 0)
        ind_1 = np.where(y == 1)

        # Find phi, mu_0, mu_1, and sigma
        phi = np.mean(y==1)
        mu_0 = np.sum(x[ind_0],0)/len(ind_0)
        mu_1 = np.sum(x[ind_1],0)/len(ind_1)
        sigma = np.zeros([len(mu_0),len(mu_0)])
        for x_i,y_i in zip(x,y):
            mu = mu_0 if y_i ==0 else mu_1
            x_i_mu = np.expand_dims((x_i - mu),1)
            sigma += np.matmul((x_i_mu),(x_i_mu).T)
        sigma = sigma/len(x)
        sigma_inv = np.linalg.inv(sigma)

        # Write theta in terms of the parameters
        self.theta0 = 0.5*(np.matmul(np.matmul(mu_0,sigma_inv),mu_0) -\
               np.matmul(np.matmul(mu_1, sigma_inv), mu_1))-\
               np.log((1-phi)/phi)

        self.theta = np.matmul(sigma_inv,mu_1) - np.matmul(sigma_inv,mu_0)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = np.matmul(self.theta.T,x).astype(float)
        return np.reciprocal(1+np.exp(-(z+self.theta0)))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
