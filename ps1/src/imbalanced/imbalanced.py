import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)

    # fit and predict
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid.T)
    np.savetxt(output_path_naive,y_valid_pred)

    # change predictions to binary
    y_valid_pred[y_valid_pred <= 0.5] = 0
    y_valid_pred[y_valid_pred > 0.5] = 1
    y_valid_pred = y_valid_pred.reshape(y_valid.shape)

    # calculate accuracies
    numCorrect = (y_valid_pred[y_valid_pred == y_valid]).shape[0]
    numEx = y_valid_pred.shape[0]
    numPos = y_valid[y_valid == 1].shape[0]
    numNeg = y_valid[y_valid == 0].shape[0]
    add = y_valid_pred + y_valid
    truePos = add[add == 2]
    trueNeg = add[add == 0]
    A = numCorrect / numEx
    A1 = truePos.shape[0] / numPos
    A0 = trueNeg.shape[0] / numNeg
    Abal = .5 * (A1 + A0)
    print("A:", A)
    print("A:", A1)
    print("A:", A0)
    print("Abal:", Abal)

    # plot decision boundary
    util.plot(x_valid, y_valid, model.theta, '6b.png')

    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)

    x_negatives = x_train[y_train==0]
    y_negatives = y_train[y_train==0]
    x_positives = x_train[y_train==1]
    y_positives = y_train[y_train==1]
    y_positives = np.repeat(y_positives, 1/kappa, axis=0)
    x_train_new = np.vstack((x_negatives, np.repeat(x_positives, 1/kappa, axis=0)))
    y_train_new = np.vstack((y_negatives, y_positives)).T
    y_train_new = np.concatenate((y_train_new[:,0], y_train_new[:,1]), axis=0)

    # fit on training and predict on validation
    model = LogisticRegression()
    model.fit(x_train_new, y_train_new)
    y_pred = model.predict(x_valid.T)
    np.savetxt(output_path_upsampling,y_valid_pred)

    # change predictions to binary
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    y_pred = y_pred.reshape(y_valid.shape)

    # calculate accuracies
    numCorrect = (y_pred[y_pred == y_valid]).shape[0]
    numEx = y_valid.shape[0]
    numPos = y_valid[y_valid == 1].shape[0]
    numNeg = y_valid[y_valid == 0].shape[0]
    add = y_pred + y_valid
    truePos = add[add == 2]
    trueNeg = add[add == 0]
    A = numCorrect / numEx
    A1 = truePos.shape[0] / numPos
    A0 = trueNeg.shape[0] / numNeg
    Abal = .5 * (A1 + A0)
    print("A:", A)
    print("A:", A1)
    print("A:", A0)
    print("Abal:", Abal)

    # plot decision boundary
    util.plot(x_valid, y_valid, model.theta, '6d.png')


    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
