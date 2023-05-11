import numpy as np
from typing import Union


def calculate_exponential_decay_probabilities(target_time_point: Union[int, float], time_points: np.ndarray,
                                              half_life: Union[float, int]) -> np.ndarray:

    """
    Calculates exponential decay probabilities for an array of time points based on a target time point and a half life.

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    target_time_point:
        The target time point.
    time points:
        The array of time points to calculate probabilities for.
    half_life:
        The half life of the exponential decay.

    Returns
    -------
    Exponential decay probabilities.

    """

    numerator = np.exp(-np.log(2) / half_life * np.clip(target_time_point - time_points, 0, np.inf))
    denominator = np.sum(numerator)

    p_t = numerator / denominator

    return p_t


def calculate_mean(x: np.ndarray, probs: Union[np.ndarray, None]=None, axis=0) -> Union[float, np.ndarray]:

    """
    Calculates (weighted) mean af a dataset x

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

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

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

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


def corr_to_cov_matrix(corr_mat: np.ndarray, vols: np.ndarray) -> np.ndarray:

    """
    Transform a covariance matrix to a correlation matrix.

    Parameters
    ----------
    corr_mat:
        Correlation matrix.
    vols:
        Volatilities.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """

    cov_mat = corr_mat * np.outer(vols, vols)

    return cov_mat


def project_logreturns_dist_params(mu_dt: np.ndarray, sigma_dt: np.ndarray, dt: int, tau: int) -> list:
    """
    Function that projects the distribution parameters of normally distributed log-returns into a given investment horizon

    Parameters
    ----------
    mu_dt
        vector of means of log-returns
    cov_mat_dt
        covariance matrix of log-returns
    dt
        time-metric of log-returns (i.e. yearly, monthly etc.)
    tau
        investment horizon

    Returns
    -------
    List
        (mu_tau, sigma_tau)
    """
    mu_tau = tau / dt * mu_dt
    sigma_tau = tau / dt * sigma_dt

    return mu_tau, sigma_tau


def calculate_mu_linear_returns(mu_tau: np.ndarray, sigma_tau: np.ndarray) -> np.ndarray:

    """
    Function that calculates the expected value of the asset prices,
    when log-returns are normally distributed

    Parameters
    ----------
    mu_tau:
        Vector of expected log returns
    sigma_tau:
        Covariance matrix of log returns

    Returns
    -------
    np.ndarray
        Vector of expected values of prices
    """

    return np.exp(mu_tau + 0.5*np.diag(sigma_tau)) - 1


def calculate_cov_mat_linear_returns(mu_tau: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:

    """
    Function that calculates the covariance matrix of the assets prices,
    when log-returns are normally distributed

    Parameters
    ----------
    mu_tau:
        Vector of expected log returns
    sigma_tau:
        Covariance matrix of log returns

    Returns
    -------
    float
        Covariance matrix of prices
    """

    mu_l = calculate_mu_linear_returns(mu_tau, cov_mat) + 1

    return np.outer(mu_l, mu_l) * (np.exp(cov_mat) - 1)


def cov_to_corr_and_vols(cov_mat: np.ndarray) -> list:

    """
    Decompose a covariance matrix to a correlation matrix and a diagonal matrix of vols

    Parameters
    ----------
    cov_mat:
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """

    vols = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / np.outer(vols, vols)
    corr_mat[corr_mat < -1], corr_mat[corr_mat > 1] = -1, 1  # numerical error

    return corr_mat, np.diag(vols)


def cov_to_corr_matrix(cov_mat: np.ndarray) -> np.ndarray:

    """
    Transform a covariance matrix to a correlation matrix.

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    cov_mat:
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """

    vols = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / np.outer(vols, vols)
    corr_mat[corr_mat < -1], corr_mat[corr_mat > 1] = -1, 1  # numerical error

    return corr_mat


def eig_decomp(corr_mat):
    D, V = np.linalg.eig(corr_mat)
    # sort eigenvalues and eigenvector in decending order
    idx = np.argsort(-D)
    D = D[idx]
    V = V[:,idx]
    D = np.diag(D)

    # control sign of eigenvectors (multiply by sign of the sum of all elements, to make sum > 0)
    V *= np.sign(np.sum(V, axis=0))

    return V, D


def calculate_mu_and_vols_pc(mu, cov_mat):

    # decompose covariance matrix
    corr_mat, vols_mat = cov_to_corr_and_vols(cov_mat)

    # eigenvalue decomposition of correlationmatrix
    P, L = eig_decomp(corr_mat)

    # calculate expected returns and vols for principalportfolios
    mu_p = P.T @ np.linalg.inv(vols_mat) @ mu
    std_p = np.diag(np.sqrt(L))

    return mu_p, std_p



