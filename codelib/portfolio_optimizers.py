import numpy as np
from scipy import optimize
from scipy.optimize import minimize

# own packages
from codelib.portfolio_measures import portfolio_variance, calculate_risk_contributions_std
from codelib.portfolio_performance_measures import calculate_sharpe_ratio, calculate_cc_ratio


def calculate_min_var_portfolio(cov_mat: np.ndarray, init_weights=None) -> np.ndarray:
    # define intial values
    n = cov_mat.shape[0]
    if init_weights is None:
        init_weights = np.repeat(1.0 / n, n)

    # define sum to one constraint
    eq_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # perform optimization
    res = optimize.minimize(lambda x: portfolio_variance(x, cov_mat),
                            init_weights,
                            method='SLSQP',
                            constraints=eq_constraint,
                            options={'ftol': 1e-9, 'disp': False},
                            bounds=[(0, 1)] * n)

    return res.x


def calculate_most_diversified_portfolio(cov_mat: np.ndarray, init_weights=None) -> np.ndarray:
    # define initial values
    n = cov_mat.shape[0]
    if init_weights is None:
        init_weights = np.repeat(1.0 / n, n)

    # define sum to one constraint
    eq_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # perform optimization
    res = optimize.minimize(lambda x: -calculate_cc_ratio(x, cov_mat), init_weights,
                            constraints=eq_constraint, bounds=[(0, 1)] * n, options={'disp': False})

    return res.x


def calculate_max_sharpe_portfolio(mu: np.ndarray, cov_mat: np.ndarray, rf: float = 0, init_weights=None) -> np.ndarray:
    # define initial values
    n = mu.shape[0]
    if init_weights is None:
        init_weights = np.repeat(1.0 / n, n)

    # define constraints: sum-to-one and long-only
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1, 'jac': lambda x: np.ones_like(x)},
                   {'type': 'ineq', 'fun': lambda x: x, 'jac': lambda x: np.eye(len(x))})

    # perform optimization
    res = optimize.minimize(lambda x: -calculate_sharpe_ratio(x, mu, cov_mat, rf), init_weights,
                            method='SLSQP',
                            constraints=constraints,
                            options={'ftol': 1e-9, 'disp': False})
    return np.abs(res.x)


def calculate_rp_portfolio(cov_mat: np.ndarray, init_weights=None) -> np.ndarray:
    # define intial values
    n = cov_mat.shape[0]
    if init_weights is None:
        init_weights = np.repeat(1.0 / n, n)

    # define sum to one constraint
    eq_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}

    # objective function
    obj_func = lambda x: np.sum(np.square(calculate_risk_contributions_std(x, cov_mat, scale=True) - 1.0 / n))

    # perform optimization
    res = optimize.minimize(obj_func,
                            init_weights,
                            constraints=[eq_constraint],
                            bounds=[(0, 1)] * n,
                            options={'ftol': 1e-16,
                                     'eps': 1e-12, 'disp': False})

    return res.x


def nrp_portfolio(cov_mat: np.ndarray) -> np.ndarray:

    vols = np.sqrt(np.diag(cov_mat))
    inv_vols = 1 / vols
    sum_inv_vols = np.sum(inv_vols)
    w = inv_vols / sum_inv_vols

    return w


def calculate_max_sharpe_pc(mu, cov_mat, r=0):

    # decompose covariance matrix
    vols = np.sqrt(np.diag(cov_mat))
    vols_mat = np.diag(vols)
    corr_mat = cov_mat / np.outer(vols, vols)
    corr_mat[corr_mat < -1], corr_mat[corr_mat > 1] = -1, 1  # numerical error

    #eigenvalue decomposition of correlationmatrix
    D, V = np.linalg.eig(corr_mat)
    # sort eigenvalues and eigenvector in decending order
    idx = D.argsort()[::-1]
    D = D[idx]
    V = V[:,idx]
    D = np.diag(D)

    # calculate expected returns and vols for principalcomponent portfolios
    mu_V = V.T @ np.linalg.inv(vols_mat) @ mu

    z = (np.linalg.inv(D) @ mu_V) / np.sum(np.linalg.inv(vols_mat) @ V @ np.linalg.inv(D) @ mu_V)
    mu_z = z.T @ np.diag(mu_V)
    var_z = (z.T @ np.sqrt(D))**2
    return z, mu_z, var_z


def tangency_portfolio(cov_mat: np.ndarray, mu: np.ndarray, rf: float) -> np.ndarray:

    """
    Calculates the (unconstrained) maximum sharpe ratio portfolio weights.

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    cov_mat:
        The covariance matrix.
    mu:
        Expected return vector.
    rf:
        The risk free rate.

    Returns
    -------
    np.ndarray
        maximum sharpe ratio portfolio weights.

    """

    num_assets = len(cov_mat)
    vec_ones = np.ones(num_assets)

    excess_mu = mu - vec_ones * rf

    cov_mat_inv = np.linalg.inv(cov_mat)

    w = cov_mat_inv @ excess_mu / (vec_ones @ cov_mat_inv @ excess_mu)

    return w


def minimum_variance_portfolio(cov_mat: np.ndarray) -> np.ndarray:

    """
    Calculates the (unconstrained) minimum-variance portfolio weights.

    (Source: codelib in repository for the course "Python for the financial economist" at CBS:
    https://github.com/staxmetrics/python_for_the_financial_economist/tree/master/codelib)

    Parameters
    ----------
    cov_mat:
        The covariance matrix.

    Returns
    -------
    np.ndarray
        Minimum variance portfolio weights.

    """

    num_assets = len(cov_mat)
    vec_ones = np.ones(num_assets)

    cov_mat_inv = np.linalg.inv(cov_mat)

    w_min_var = cov_mat_inv @ vec_ones / (vec_ones @ cov_mat_inv @ vec_ones)

    return w_min_var
