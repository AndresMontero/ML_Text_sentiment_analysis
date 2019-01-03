# -*- coding: utf-8 -*-
"""some helper functions for project 2."""
import csv
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

#### ----------------- taken from project 1 implementations ----------------####
def batch_iter(y, tx, batch_size, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        batch_size (int): Size of the batch
        num_batches (int, optional): Defaults to 1. Number of batches
        shuffle (bool, optional): Defaults to True. Shuffle or not
    """

    data_size = len(y)
    num_batches = math.floor(data_size / batch_size)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


#### --------------- taken from project 1 helpers ---------------------####
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def standardize(x):
    """Standardize the original data set.

    Args:
        x (numpy.array): Array with data for x

    Returns:
        (tuple): tuple containing:

            x (numpy.array): Standardized array
            mean_x (numpy.array): Arithmetic Mean
            std_x (numpy.array): Standard deviation
    """

    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def standardize_test(x_test, mean, std):
    """Standardize the values of testing_x depending on the values of mean and std of the training x vector

    Args:
        x_test (numpy.array): Testing x values
        mean (numpy.array): Mean values
        std (numpy.array): Standard deviation values

    Returns:
        numpy.array: Standarized X
    """

    new_x = x_test.copy()
    new_x = (new_x - mean) / std
    return new_x


#### ----------------------- new methods --------------------####
def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / len(labels)


def shuffle(x_train, y_train):
        np.random.seed(4133)
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        return x_train[shuffle_indices, :], y_train[shuffle_indices]

def add_offset(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)


def plot_val_acc(stats):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(stats[:, 0], 'g-')
    ax2.plot(stats[:, 1], 'b-')

    ax1.set_xlabel('Sampling Steps')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Accuracy', color='b')

    plt.show()
