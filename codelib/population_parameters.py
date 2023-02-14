import numpy as np
from typing import Union


def beregn_middelvaerdi(x: np.ndarray, probs: Union[np.ndarray, None] = None, axis=0) -> Union[float, np.ndarray]:
    """
    Beregn (vægtet) middelværdi af et datasæt x

    Parameters
    ----------
    x:
        Datasæt
    probs:
        Udfaldssandsynligheder
    axis:
        Akse for hvilken middelvaerdien beregnes over

    Returns
    -------
    Union[float, np.ndarray]
        Middelvaerdi

    """

    m = np.average(x, weights=probs, axis=axis)
    return m

def beregn_kovarians_mat(x: np.ndarray, probs: Union[np.ndarray, None] = None, axis = 0) -> Union[float, np.ndarray]:
    """
    Beregn (vægtet) kovarains matrice af et datasæt x

    Parameters
    ----------
    x:
        Datasæt
    probs:
        Udfaldssandsynligheder
    axis:
        Akse for hvilken covarians matricen beregnes over

    Returns
    -------
    Union[float, np.ndarray]
        Kovarians matrice

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