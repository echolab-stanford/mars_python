"""
Measurement Error Corrected Least Squares (ME-LS)

This module implements classical measurement error corrected least squares
estimation with cluster-robust standard errors. The core idea is to adjust
the normal equations by subtracting the measurement error covariance:

    beta_hat = solve((X'X) - (n * Sigma)) * (X'Y)

where Sigma is the measurement error covariance matrix.
"""

from typing import Dict, List, Union, Optional, Tuple
import numpy as np
from scipy import stats, linalg
import pandas as pd
import warnings
from tqdm import tqdm


def _create_meat_matrix(
    cluster: np.ndarray,
    X: np.ndarray, 
    Y: np.ndarray,
    Sigma: Union[np.ndarray, List[np.ndarray]],
    beta_hat: np.ndarray,
    is_inid: bool = False
) -> np.ndarray:
    """
    Helper function to create meat matrix for sandwich variance estimator.
    
    Parameters
    ----------
    cluster : np.ndarray
        Cluster identifiers
    X : np.ndarray
        Design matrix
    Y : np.ndarray  
        Outcome vector or matrix
    Sigma : np.ndarray or List[np.ndarray]
        Measurement error covariance (single matrix for iid, list for inid)
    beta_hat : np.ndarray
        Parameter estimates (K x p matrix for p outcomes)
    is_inid : bool
        Whether this is the inid (heteroskedastic) case
        
    Returns
    -------
    np.ndarray
        Meat matrix for sandwich estimator
    """
    K = X.shape[1]
    
    # Handle multiple outcomes
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if beta_hat.ndim == 1:
        beta_hat = beta_hat.reshape(-1, 1)
        
    p = Y.shape[1]  # Number of outcomes
    meat = np.zeros((K, K))
    clusters = np.unique(cluster)
    
    # Use progress bar for large number of clusters
    for g in tqdm(clusters, desc="Computing cluster contributions"):
        indices = cluster == g
        X_g = X[indices]
        Y_g = Y[indices]
        
        # Initialize moment for this cluster
        moment = np.zeros((K, 1))
        
        for i, idx in enumerate(np.where(indices)[0]):
            # For multiple outcomes, we need to sum over outcomes
            for j in range(p):
                # Residual for observation i in cluster g, outcome j
                residual = Y_g[i, j] - X_g[i] @ beta_hat[:, j]
                
                # Measurement error adjustment
                if is_inid:
                    # Individual-specific Sigma matrix
                    me_adjustment = Sigma[idx] @ beta_hat[:, j]
                else:
                    # Common Sigma matrix  
                    me_adjustment = Sigma @ beta_hat[:, j]
                
                # Add to cluster moment
                moment += (X_g[i:i+1].T * residual + me_adjustment.reshape(-1, 1))
        
        # Add outer product to meat matrix
        meat += moment @ moment.T
    
    return meat


def _apply_eigenvalue_correction(vcov_matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Apply eigenvalue correction to ensure positive semi-definiteness.
    
    Parameters
    ----------
    vcov_matrix : np.ndarray
        Variance-covariance matrix to correct
    tol : float
        Minimum eigenvalue tolerance
        
    Returns
    -------
    np.ndarray
        Corrected variance-covariance matrix
    """
    eigenvals, eigenvecs = linalg.eigh(vcov_matrix)
    corrected_vals = np.maximum(eigenvals, tol)
    return eigenvecs @ np.diag(corrected_vals) @ eigenvecs.T


def me_ls_iid(
    X: np.ndarray,
    Y: np.ndarray, 
    Sigma: np.ndarray,
    cluster1: Optional[np.ndarray] = None,
    cluster2: Optional[np.ndarray] = None,
    eigfix: bool = False,
    tol: float = 1e-8,
    level: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Measurement Error Corrected Least Squares with iid errors
    
    Computes classical measurement error corrected least squares estimates
    with cluster-robust standard errors for identically distributed ME.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    Y : np.ndarray
        Outcome vector (n x 1) or matrix (n x p)
    Sigma : np.ndarray
        Measurement error covariance matrix (k x k)
    cluster1 : np.ndarray, optional
        Primary clustering variable
    cluster2 : np.ndarray, optional  
        Secondary clustering variable for two-way clustering
    eigfix : bool, default=False
        Whether to apply eigenvalue correction
    tol : float, default=1e-8
        Tolerance for eigenvalue correction
    level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'coefficients': Point estimates
        - 'vcov': Variance-covariance matrix
        - 'robust_se': Cluster-robust standard errors
        - 'p_values': Two-sided p-values
        - 'ci_lower': Lower confidence bounds
        - 'ci_upper': Upper confidence bounds
        
    Examples
    --------
    >>> result = me_ls_iid(
    ...     X=design_matrix,
    ...     Y=outcomes,
    ...     Sigma=measurement_error_cov,
    ...     cluster1=cluster_ids
    ... )
    >>> print("Coefficients:", result['coefficients'])
    >>> print("Standard errors:", result['robust_se'])
    """
    # Input validation and setup
    X = np.asarray(X)
    Y = np.asarray(Y)
    Sigma = np.asarray(Sigma)
    
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n, K = X.shape
    
    if cluster1 is None:
        cluster1 = np.arange(n)
    else:
        cluster1 = np.asarray(cluster1)
    
    if cluster2 is not None:
        cluster2 = np.asarray(cluster2)
    
    # Check dimensions
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if Sigma.shape != (K, K):
        raise ValueError(f"Sigma must be {K} x {K} matrix")
    
    # Compute point estimates with measurement error correction
    try:
        XTX_corrected = X.T @ X - n * Sigma
        
        # Check for singularity by computing condition number
        cond_num = np.linalg.cond(XTX_corrected)
        if cond_num > 1e15:  # Effectively singular
            raise ValueError("Singular matrix after measurement error correction")
            
        XTY = X.T @ Y
        beta_hat = linalg.solve(XTX_corrected, XTY)
    except linalg.LinAlgError:
        raise ValueError("Singular matrix after measurement error correction")
    
    # Bread matrix (inverse of corrected information)
    try:
        bread = linalg.inv(XTX_corrected)
    except linalg.LinAlgError:
        raise ValueError("Cannot invert corrected information matrix")
    
    # Compute variance-covariance matrix
    if cluster2 is None:
        # One-way clustering
        meat = _create_meat_matrix(cluster1, X, Y, Sigma, beta_hat, is_inid=False)
        vcov_matrix = bread @ meat @ bread
    else:
        # Two-way clustering using inclusion-exclusion principle
        meat1 = _create_meat_matrix(cluster1, X, Y, Sigma, beta_hat, is_inid=False)
        meat2 = _create_meat_matrix(cluster2, X, Y, Sigma, beta_hat, is_inid=False)
        
        # Create interaction clusters 
        cluster12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(cluster1, cluster2)])
        meat12 = _create_meat_matrix(cluster12, X, Y, Sigma, beta_hat, is_inid=False)
        
        # Two-way clustering formula
        vcov_matrix = bread @ (meat1 + meat2 - meat12) @ bread
    
    # Apply eigenvalue correction if requested
    if eigfix:
        vcov_matrix = _apply_eigenvalue_correction(vcov_matrix, tol)
    
    # Extract results for each outcome (if multiple)
    results = {}
    for j in range(Y.shape[1]):
        # Standard errors
        robust_se = np.sqrt(np.diag(vcov_matrix))
        
        # P-values (two-sided)
        t_stats = beta_hat[:, j] / robust_se
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # Confidence intervals
        ci_width = stats.norm.ppf((1 + level) / 2) * robust_se
        ci_lower = beta_hat[:, j] - ci_width
        ci_upper = beta_hat[:, j] + ci_width
        
        if Y.shape[1] == 1:
            # Single outcome case
            results = {
                'coefficients': beta_hat[:, 0],
                'vcov': vcov_matrix,
                'robust_se': robust_se,
                'p_values': p_values,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        else:
            # Multiple outcomes case
            results[f'outcome_{j}'] = {
                'coefficients': beta_hat[:, j],
                'vcov': vcov_matrix,
                'robust_se': robust_se,
                'p_values': p_values,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
    
    return results


def me_ls_inid(
    X: np.ndarray,
    Y: np.ndarray,
    Sigma: List[np.ndarray],
    cluster1: Optional[np.ndarray] = None,
    cluster2: Optional[np.ndarray] = None,
    eigfix: bool = False,
    tol: float = 1e-8,
    level: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Measurement Error Corrected Least Squares with heteroskedastic errors
    
    Computes classical measurement error corrected least squares estimates  
    with cluster-robust standard errors for independent (but not identically 
    distributed) measurement error.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n x k)
    Y : np.ndarray
        Outcome vector (n x 1) or matrix (n x p)
    Sigma : List[np.ndarray]
        List of measurement error covariance matrices, one per observation
    cluster1 : np.ndarray, optional
        Primary clustering variable
    cluster2 : np.ndarray, optional
        Secondary clustering variable for two-way clustering
    eigfix : bool, default=False
        Whether to apply eigenvalue correction
    tol : float, default=1e-8
        Tolerance for eigenvalue correction
    level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with same structure as me_ls_iid()
        
    Examples
    --------
    >>> # Create observation-specific error covariances
    >>> sigma_list = [np.eye(k) * error_var[i] for i in range(n)]
    >>> result = me_ls_inid(
    ...     X=design_matrix,
    ...     Y=outcomes,
    ...     Sigma=sigma_list,
    ...     cluster1=cluster_ids
    ... )
    """
    # Input validation
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n, K = X.shape
    
    if len(Sigma) != n:
        raise ValueError(f"Sigma list must have {n} matrices, one per observation")
    
    # Validate that all Sigma matrices have correct dimensions
    for i, S in enumerate(Sigma):
        if S.shape != (K, K):
            raise ValueError(f"Sigma[{i}] must be {K} x {K} matrix")
    
    if cluster1 is None:
        cluster1 = np.arange(n)
    else:
        cluster1 = np.asarray(cluster1)
    
    if cluster2 is not None:
        cluster2 = np.asarray(cluster2)
    
    # Compute average Sigma matrix
    Sigma_bar = np.mean(Sigma, axis=0)
    
    # Compute point estimates with average measurement error correction
    try:
        XTX_corrected = X.T @ X - n * Sigma_bar
        XTY = X.T @ Y
        beta_hat = linalg.solve(XTX_corrected, XTY)
    except linalg.LinAlgError:
        raise ValueError("Singular matrix after measurement error correction")
    
    # Bread matrix
    try:
        bread = linalg.inv(XTX_corrected)
    except linalg.LinAlgError:
        raise ValueError("Cannot invert corrected information matrix")
    
    # Compute variance-covariance matrix
    if cluster2 is None:
        # One-way clustering
        meat = _create_meat_matrix(cluster1, X, Y, Sigma, beta_hat, is_inid=True)
        vcov_matrix = bread @ meat @ bread
    else:
        # Two-way clustering
        meat1 = _create_meat_matrix(cluster1, X, Y, Sigma, beta_hat, is_inid=True)
        meat2 = _create_meat_matrix(cluster2, X, Y, Sigma, beta_hat, is_inid=True)
        
        # Create interaction clusters
        cluster12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(cluster1, cluster2)])
        meat12 = _create_meat_matrix(cluster12, X, Y, Sigma, beta_hat, is_inid=True)
        
        # Two-way clustering formula
        vcov_matrix = bread @ (meat1 + meat2 - meat12) @ bread
    
    # Apply eigenvalue correction if requested
    if eigfix:
        vcov_matrix = _apply_eigenvalue_correction(vcov_matrix, tol)
    
    # Extract results
    results = {}
    for j in range(Y.shape[1]):
        # Standard errors
        robust_se = np.sqrt(np.diag(vcov_matrix))
        
        # P-values (two-sided)
        t_stats = beta_hat[:, j] / robust_se
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # Confidence intervals
        ci_width = stats.norm.ppf((1 + level) / 2) * robust_se
        ci_lower = beta_hat[:, j] - ci_width
        ci_upper = beta_hat[:, j] + ci_width
        
        if Y.shape[1] == 1:
            # Single outcome case
            results = {
                'coefficients': beta_hat[:, 0],
                'vcov': vcov_matrix,
                'robust_se': robust_se,
                'p_values': p_values,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        else:
            # Multiple outcomes case  
            results[f'outcome_{j}'] = {
                'coefficients': beta_hat[:, j],
                'vcov': vcov_matrix,
                'robust_se': robust_se,
                'p_values': p_values,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
    
    return results