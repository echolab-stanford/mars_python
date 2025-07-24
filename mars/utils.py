"""
Utility functions for MARS statistical computations

This module provides helper functions for common operations used throughout
the MARS framework, including data validation, matrix operations, and 
statistical utilities.
"""

import numpy as np
from scipy import stats
from typing import Union, Optional, Tuple, List
import warnings


def validate_mars_inputs(
    M: np.ndarray,
    A: np.ndarray, 
    hat_mu: np.ndarray,
    pi: np.ndarray,
    allow_missing_M: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and standardize inputs for MARS estimators.
    
    Parameters
    ----------
    M : np.ndarray
        Ground truth values
    A : np.ndarray  
        Annotation indicators
    hat_mu : np.ndarray
        ML predictions
    pi : np.ndarray
        Annotation probabilities
    allow_missing_M : bool, default=True
        Whether to allow M to have missing values where A=0
        
    Returns
    -------
    Tuple[np.ndarray, ...]
        Validated and standardized input arrays
        
    Raises
    ------
    ValueError
        If inputs fail validation checks
    """
    # Convert to numpy arrays
    M, A, hat_mu, pi = map(np.asarray, [M, A, hat_mu, pi])
    
    # Check lengths match
    if not (len(M) == len(A) == len(hat_mu) == len(pi)):
        raise ValueError("All input arrays must have the same length")
    
    # Check annotation indicators are binary
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Annotation indicators A must be 0 or 1")
    
    # Check annotation probabilities are positive
    if np.any(pi <= 0):
        raise ValueError("All annotation probabilities must be positive")
    
    if np.any(pi > 1):
        warnings.warn("Some annotation probabilities > 1, unusual but allowed")
    
    # Check ground truth values where annotated
    if not allow_missing_M:
        if np.any(np.isnan(M)):
            raise ValueError("Ground truth values M cannot contain NaN")
    else:
        # Check that M is non-missing where A=1
        annotated_idx = A == 1
        if np.any(np.isnan(M[annotated_idx])):
            raise ValueError("Ground truth values M cannot be missing where A=1")
    
    return M, A, hat_mu, pi


def compute_pseudo_outcomes(
    M: np.ndarray,
    A: np.ndarray,
    hat_mu: np.ndarray, 
    pi: np.ndarray
) -> np.ndarray:
    """
    Compute pseudo-outcomes for MARS estimators.
    
    The pseudo-outcome is the key quantity in MARS estimation:
        pseudo = hat_mu + (A / pi) * (M - hat_mu)
    
    Parameters
    ----------
    M : np.ndarray
        Ground truth values
    A : np.ndarray
        Annotation indicators  
    hat_mu : np.ndarray
        ML predictions
    pi : np.ndarray
        Annotation probabilities
        
    Returns
    -------
    np.ndarray
        Pseudo-outcomes
    """
    M, A, hat_mu, pi = validate_mars_inputs(M, A, hat_mu, pi)
    return hat_mu + (A / pi) * (M - hat_mu)


def compute_influence_function(
    pseudo: np.ndarray,
    theta_hat: float,
    scale: float = 1.0
) -> np.ndarray:
    """
    Compute influence function for asymptotic variance.
    
    Parameters
    ----------
    pseudo : np.ndarray
        Pseudo-outcomes
    theta_hat : float
        Point estimate
    scale : float, default=1.0
        Scaling factor
        
    Returns
    -------
    np.ndarray
        Influence function values
    """
    return scale * (pseudo - theta_hat)


def robust_matrix_inverse(
    matrix: np.ndarray,
    regularization: float = 1e-12,
    method: str = 'cholesky'
) -> np.ndarray:
    """
    Compute robust matrix inverse with regularization.
    
    Parameters
    ---------- 
    matrix : np.ndarray
        Matrix to invert
    regularization : float, default=1e-12
        Regularization parameter for numerical stability
    method : str, default='cholesky'
        Method to use ('cholesky', 'svd', 'eig')
        
    Returns
    -------
    np.ndarray
        Inverse matrix
        
    Raises
    ------
    ValueError
        If matrix is not invertible
    """
    try:
        if method == 'cholesky':
            # Fast for positive definite matrices
            L = np.linalg.cholesky(matrix + regularization * np.eye(matrix.shape[0]))
            return np.linalg.inv(L.T) @ np.linalg.inv(L)
        elif method == 'svd':
            # Most robust but slower
            U, s, Vt = np.linalg.svd(matrix)
            s_inv = np.where(s > regularization, 1/s, 0)
            return Vt.T @ np.diag(s_inv) @ U.T
        elif method == 'eig':
            # Middle ground
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals_inv = np.where(eigenvals > regularization, 1/eigenvals, 0)
            return eigenvecs @ np.diag(eigenvals_inv) @ eigenvecs.T
        else:
            # Standard inversion with regularization
            return np.linalg.inv(matrix + regularization * np.eye(matrix.shape[0]))
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not invertible even with regularization")


def cluster_summary_stats(
    data: np.ndarray,
    clusters: np.ndarray,
    statistic: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute summary statistics by cluster.
    
    Parameters
    ----------
    data : np.ndarray
        Data values
    clusters : np.ndarray
        Cluster identifiers
    statistic : str, default='mean'
        Statistic to compute ('mean', 'sum', 'count', 'var')
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (unique_clusters, statistics)
    """
    unique_clusters = np.unique(clusters)
    stats_by_cluster = []
    
    for cluster in unique_clusters:
        cluster_data = data[clusters == cluster]
        if statistic == 'mean':
            stat = np.mean(cluster_data)
        elif statistic == 'sum':
            stat = np.sum(cluster_data)
        elif statistic == 'count':
            stat = len(cluster_data)
        elif statistic == 'var':
            stat = np.var(cluster_data, ddof=1) if len(cluster_data) > 1 else 0
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        stats_by_cluster.append(stat)
    
    return unique_clusters, np.array(stats_by_cluster)


def format_estimation_results(
    result: dict,
    param_names: Optional[List[str]] = None,
    digits: int = 4
) -> str:
    """
    Format estimation results for pretty printing.
    
    Parameters
    ----------
    result : dict
        Results dictionary from MARS estimator
    param_names : List[str], optional
        Parameter names for display
    digits : int, default=4
        Number of decimal places
        
    Returns
    -------
    str
        Formatted results string
    """
    if 'est' in result:
        # Single parameter case
        output = f"Estimate: {result['est']:.{digits}f}\n"
        output += f"Std Error: {result['se']:.{digits}f}\n" 
        output += f"95% CI: [{result['ci_lower']:.{digits}f}, {result['ci_upper']:.{digits}f}]\n"
        return output
    else:
        # Multiple parameters case
        output = "Parameter Estimates:\n"
        output += "-" * 50 + "\n"
        
        coeffs = result.get('coefficients', [])
        ses = result.get('robust_se', [])
        cis_lower = result.get('ci_lower', [])
        cis_upper = result.get('ci_upper', [])
        
        for i, (coef, se, ci_l, ci_u) in enumerate(zip(coeffs, ses, cis_lower, cis_upper)):
            param_name = param_names[i] if param_names else f"param_{i}"
            output += f"{param_name:12}: {coef:8.{digits}f} ({se:6.{digits}f}) "
            output += f"[{ci_l:7.{digits}f}, {ci_u:7.{digits}f}]\n"
        
        return output


def check_identification_conditions(
    A: np.ndarray,
    pi: np.ndarray,
    min_annotation_rate: float = 0.01,
    min_pi: float = 0.001
) -> List[str]:
    """
    Check identification conditions for MARS estimators.
    
    Parameters
    ----------
    A : np.ndarray
        Annotation indicators
    pi : np.ndarray  
        Annotation probabilities
    min_annotation_rate : float, default=0.01
        Minimum required annotation rate
    min_pi : float, default=0.001
        Minimum required annotation probability
        
    Returns
    -------
    List[str]
        List of warning messages (empty if all conditions satisfied)
    """
    warnings = []
    
    # Check annotation rate
    annotation_rate = np.mean(A)
    if annotation_rate < min_annotation_rate:
        warnings.append(f"Low annotation rate: {annotation_rate:.1%} < {min_annotation_rate:.1%}")
    
    # Check minimum annotation probabilities
    min_observed_pi = np.min(pi)
    if min_observed_pi < min_pi:
        warnings.append(f"Very low annotation probabilities: min = {min_observed_pi:.6f}")
    
    # Check for extreme annotation probabilities
    max_pi = np.max(pi)
    if max_pi > 10:
        warnings.append(f"Very high annotation probabilities: max = {max_pi:.2f}")
    
    # Check variation in annotation probabilities
    pi_var = np.var(pi)
    if pi_var < 1e-10:
        warnings.append("No variation in annotation probabilities")
    
    return warnings


def simulate_mars_data(
    n: int,
    k: int = 1,
    annotation_rate: float = 0.3,
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> dict:
    """
    Simulate data for testing MARS estimators.
    
    Parameters
    ----------
    n : int
        Sample size
    k : int, default=1
        Number of covariates
    annotation_rate : float, default=0.3
        Proportion of observations to annotate
    noise_level : float, default=0.1
        Noise level in ML predictions
    random_state : int, optional
        Random seed
        
    Returns
    -------
    dict
        Dictionary containing simulated data arrays
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # True outcome
    X = np.random.randn(n, k)
    true_beta = np.random.randn(k)
    Y_true = X @ true_beta + np.random.randn(n) * 0.5
    
    # ML predictions with noise
    hat_mu = Y_true + np.random.randn(n) * noise_level
    
    # Annotation process
    pi = np.random.uniform(0.1, 0.5, n)  # Varying annotation probabilities
    A = np.random.binomial(1, pi * annotation_rate / np.mean(pi))
    
    # Observed ground truth (only where annotated)
    M = np.where(A == 1, Y_true, 0)
    
    return {
        'Y_true': Y_true,
        'X': X,
        'M': M,
        'A': A,
        'hat_mu': hat_mu,
        'pi': pi,
        'true_beta': true_beta
    }