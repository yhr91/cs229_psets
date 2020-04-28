import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_eval.T)

    # Plot decision boundary on top of validation set set
    util.plot(x_eval,y_eval,clf.theta,save_path.split('.')[0]+'.jpeg')

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path,predictions)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=np.zeros([3,1]), verbose=True):
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

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        itr = 0
        while(itr<self.max_iter):

            # Calculate the Hessian
            H = np.zeros([len(x[0]),len(x[0])])
            dJ = np.zeros([len(x[0]),1])

            for x_i,y_i in zip(x,y):
                x_i = np.expand_dims(x_i,1)

                g1 = self.predict(x_i)
                xxT= np.matmul(x_i,x_i.T)

                H += g1*(1-g1) * xxT   # Hessian
                dJ += -(y_i - g1) * x_i  # Gradient

            H = H/len(x)
            dJ = dJ/len(x)

            update = self.step_size * np.matmul(np.linalg.inv(H),dJ)
            self.theta = self.theta - update

            if (self.verbose):
                print(np.linalg.norm(update))

            if np.linalg.norm(update)<self.eps:
                break
            else:
                itr += 1

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Calculate the sigmoid function
        z = np.matmul(self.theta.T,x).astype(float)
        return np.reciprocal(1+np.exp(-z))

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
