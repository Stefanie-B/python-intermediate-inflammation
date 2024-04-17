"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
from functools import reduce


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    :returns: ndarray
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """ Calculate the daily mean of a 2D inflammation data array.

    :param data: 2D array of inflammation data with patients across axis 0 and time across axis 1
    :returns: Mean values measured on each day
    """
    return np.mean(data, axis=0)


def daily_stddev(data):
    """ Calculate the daily standard deviation of a 2D inflammation data array.

    :param data: 2D array of inflammation data with patients across axis 0 and time across axis 1
    :returns: standard deviation values measured on each day
    """
    return np.std(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: 2D array of inflammation data with patients across axis 0 and time across axis 1
    :returns: Maximum values measured on each day
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: 2D array of inflammation data with patients across axis 0 and time across axis 1
    :returns: Minimum values measured on each day
    """
    return np.min(data, axis=0)


def daily_above_threshold(patient_num, data, threshold):
    """Determine the total number of days the inflammation value exceeds a given threshold for a given patient.

    :param patient_num: The patient row number
    :param data: A 2D data array with inflammation data
    :param threshold: An inflammation threshold to check each daily value against
    :returns: A boolean list representing whether each patient's daily inflammation exceeded the threshold
    """

    return reduce(lambda a, b: int(a > threshold) + b, data[patient_num], 0)


def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.
    
    Negative values are rounded to 0.

    :param data: ndarray patient data
    :return: normalized ndarray, output shape?????
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('Data must be a numpy array')
    if len(data.shape) != 2:
        raise TypeError('Data must be a 2D array')
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')
    data_max = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / data_max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised
