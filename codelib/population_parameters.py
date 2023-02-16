import numpy as np
from typing import Union


def calculate_mean(x: np.ndarray, probs: Union[np.ndarray, None]=None, axis=0) -> Union[float, np.ndarray]:

    """
    Calculates (weighted) mean af a datasæt x

    Parameters
    ----------
    x:
        Data to calculate mean for
    probs:
        Probabilities
    axis:
        Axis over which to calculate over

    Returns
    -------
    Union[float, np.ndarray]
        Mean

    """

    m = np.average(x, weights=probs, axis=axis)

    return m

def calculate_cov_mat(x: np.ndarray, probs: np.ndarray, axis: int = 0) -> np.ndarray:

    """
    Estimates a covariance matrix based on a historical dataset and a set of probabilities.

    Parameters
    ----------
    x:
        The dataset to estimate covariance for.
    probs:
        The probabilities to weight the observations of the dataset by.
    axis:
        The axis to estimate over.

    Returns
    -------
    np.ndarray
        The estimated covariance matrix.

    """

    x = x.T if axis == 1 else x

    expected_x_squared = np.sum(probs[:, None, None] * np.einsum('ji, jk -> jik', x, x), axis=0)
    mu = probs @ x
    mu_squared = np.einsum('j, i -> ji', mu, mu)
    cov_mat = expected_x_squared - mu_squared

    return cov_mat

def calculate_variance(x: np.ndarray, probs: Union[np.ndarray, None] = None, axis=0) -> Union[float, np.ndarray]:

    """
    Calculates variance.

    Parameters
    ----------
    x:
        Data to calculate variance for.
    probs:
        Probabilities.
    axis:
        Axis over which to calculate.

    Returns
    -------
    Union[float, np.ndarray]
        Variance.

    """

    m = np.average(x, weights=probs, axis=axis)

    return np.average(np.square(x - m), weights=probs)


# FUNCTIONS:
# other estimators later ie. exponential decay

#FUNCTIONS:

# calculate rolling mean returns
# calculate rolling cov matrices
# calculate rolling volatilities
# other relevant input parameters?


