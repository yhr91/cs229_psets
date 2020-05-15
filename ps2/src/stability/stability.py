# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    loss = np.sum((Y - probs) ** 2)
    grad = (Y - probs).dot(X)

    return grad, loss, probs


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    thetas = []
    losses = []
    accuracies = []
    avg_correct_scores = []

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad, loss, scores = calc_grad(X, Y, theta)
        thetas.append((i,np.sum(theta)))
        theta = theta + learning_rate * grad
        
        probs = np.copy(scores)
        probs[probs >= 0.5] = 1
        probs[probs < 0.5] = 0
        accuracy = np.sum(probs==Y)/probs.shape[0]
        avg_correct_score = np.sum(scores[probs==1])/scores[probs==1].shape[0]
        accuracies.append((i,accuracy))

        if i % 1000 == 0:
            print('Finished %d iterations' % i)
            #thetas.append((i,np.sum(theta)))
            losses.append((i,loss))
            print(accuracy)
            avg_correct_scores.append((i,avg_correct_score))

            if i >= 1000000:
                break
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return thetas, losses, accuracies, theta, avg_correct_scores


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    a1, a2, a3, a4, a5 =  logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    b1, b2, b3, b4, b5 = logistic_regression(Xb, Yb)
    return a1, a2, a3, a4, a5, b1, b2, b3, b4, b5


if __name__ == '__main__':
    main()
