"""
MARS Estimators

Core statistical estimators for the MARS framework that combine machine learning
predictions with partial human annotations to provide robust inference.

All estimators use influence functions and pseudo-outcomes of the form:
    pseudo = hat_mu + (A / pi) * (M - hat_mu)

where:
- hat_mu: ML predictions for all observations
- A: annotation indicators (1 if annotated, 0 otherwise)  
- pi: annotation probabilities
- M: ground truth values (only non-zero where A=1)
"""

from typing import Dict, List, Union, Optional
import numpy as np
from scipy import stats
import warnings


def mars_mean_estimator(
    M: np.ndarray,
    A: np.ndarray, 
    hat_mu: np.ndarray,
    pi: np.ndarray,
    scale: float = 1.0,
    level: float = 0.95
) -> Dict[str, float]:
    """
    MARS Mean Estimator
    
    Computes MARS mean estimates and confidence intervals for population means
    using partially annotated data.
    
    Parameters
    ----------
    M : np.ndarray
        Ground truth values (with non-zero/non-missing entries indicated by A)
    A : np.ndarray
        Annotation indicators (1 if annotated, 0 otherwise)
    hat_mu : np.ndarray
        ML predictions for all observations
    pi : np.ndarray
        Annotation probabilities
    scale : float, default=1.0
        Scaling factor for the estimate
    level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'est': Point estimate
        - 'se': Standard error
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        
    Examples
    --------
    >>> result = mars_mean_estimator(
    ...     M=ground_truth_values,
    ...     A=annotation_indicators, 
    ...     hat_mu=ml_predictions,
    ...     pi=annotation_probs
    ... )
    >>> print(f"Estimate: {result['est']:.3f} ± {result['se']:.3f}")
    """
    # Input validation
    M, A, hat_mu, pi = map(np.asarray, [M, A, hat_mu, pi])
    
    if not (len(M) == len(A) == len(hat_mu) == len(pi)):
        raise ValueError("All input arrays must have the same length")
    
    if np.any(pi <= 0):
        raise ValueError("All annotation probabilities must be positive")
    
    # Sample size
    I = len(hat_mu)
    
    # Compute pseudo-outcomes using influence function
    pseudo = hat_mu + (A / pi) * (M - hat_mu)
    
    # Point estimate (unscaled)
    hat_theta_unscaled = np.mean(pseudo)
    
    # Point estimate (scaled)
    hat_theta = hat_theta_unscaled * scale
    
    # Compute asymptotic variance (using unscaled estimate)
    psi = pseudo - hat_theta_unscaled
    avar = np.mean(psi**2) * (scale**2)
    
    # Standard error
    se = np.sqrt(avar / I)
    
    # Confidence interval
    ci_width = stats.norm.ppf((1 + level) / 2) * se
    
    return {
        'est': hat_theta,
        'se': se, 
        'ci_lower': hat_theta - ci_width,
        'ci_upper': hat_theta + ci_width
    }


def mars_linreg_estimator(
    C_resid: np.ndarray,
    A: np.ndarray,
    M: np.ndarray, 
    hat_mu: np.ndarray,
    pi: np.ndarray,
    level: float = 0.95
) -> Dict[str, float]:
    """
    MARS Linear Regression Estimator
    
    Computes MARS linear regression coefficient estimates and confidence intervals
    using residualized covariates and pseudo-outcomes.
    
    Parameters
    ----------
    C_resid : np.ndarray
        Residualized covariates (must be mean-zero)
    A : np.ndarray
        Annotation indicators  
    M : np.ndarray
        Ground truth outcome values
    hat_mu : np.ndarray
        ML predictions
    pi : np.ndarray
        Annotation probabilities
    level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, float]
        Dictionary with estimate, standard error, and confidence interval
        
    Examples
    --------
    >>> result = mars_linreg_estimator(
    ...     C_resid=residualized_covariates,
    ...     A=annotation_indicators,
    ...     M=ground_truth_outcomes, 
    ...     hat_mu=ml_predictions,
    ...     pi=annotation_probs
    ... )
    """
    # Input validation
    arrays = [C_resid, A, M, hat_mu, pi]
    C_resid, A, M, hat_mu, pi = map(np.asarray, arrays)
    
    if not all(len(arr) == len(C_resid) for arr in arrays):
        raise ValueError("All input arrays must have the same length")
    
    # Check that C_resid is mean-zero (with tolerance)
    if abs(np.mean(C_resid)) > 1e-10:
        raise ValueError("C_resid must be mean-zero (residualized)")
    
    if np.any(pi <= 0):
        raise ValueError("All annotation probabilities must be positive")
    
    # Sample size
    I = len(C_resid)
    
    # Compute pseudo-outcomes
    pseudo = hat_mu + (A / pi) * (M - hat_mu)
    
    # Compute numerator and denominator estimates  
    hat_theta_num = np.mean(pseudo * C_resid)
    hat_theta_denom = np.mean(C_resid**2)
    
    # Point estimate
    hat_theta = hat_theta_num / hat_theta_denom
    
    # Compute asymptotic variance using influence function
    psi = (1 / hat_theta_denom) * C_resid * (pseudo - C_resid * hat_theta)
    avar = np.mean(psi**2)
    
    # Standard error
    se = np.sqrt(avar / I)
    
    # Confidence interval
    ci_width = stats.norm.ppf((1 + level) / 2) * se
    
    return {
        'est': hat_theta,
        'se': se,
        'ci_lower': hat_theta - ci_width, 
        'ci_upper': hat_theta + ci_width
    }


def mars_liv_estimator(
    Z_resid: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    hat_mu: np.ndarray, 
    pi: np.ndarray,
    level: float = 0.95
) -> Dict[str, float]:
    """
    MARS Linear IV Estimator
    
    Computes MARS linear instrumental variables estimates for constant treatment
    effects using residualized instruments and pseudo-outcomes.
    
    Parameters
    ----------
    Z_resid : np.ndarray
        Residualized instrument values
    Y : np.ndarray  
        Outcome values
    M : np.ndarray
        Ground truth treatment values
    A : np.ndarray
        Annotation indicators
    hat_mu : np.ndarray
        ML predictions for treatment
    pi : np.ndarray
        Annotation probabilities
    level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, float]
        Dictionary with IV estimate, standard error, and confidence interval
        
    Examples
    --------
    >>> result = mars_liv_estimator(
    ...     Z_resid=residualized_instruments,
    ...     Y=outcomes,
    ...     M=ground_truth_treatment,
    ...     A=annotation_indicators,
    ...     hat_mu=ml_treatment_predictions,
    ...     pi=annotation_probs
    ... )
    """
    # Input validation
    arrays = [Z_resid, Y, M, A, hat_mu, pi]
    Z_resid, Y, M, A, hat_mu, pi = map(np.asarray, arrays)
    
    if not all(len(arr) == len(Z_resid) for arr in arrays):
        raise ValueError("All input arrays must have the same length")
        
    if np.any(pi <= 0):
        raise ValueError("All annotation probabilities must be positive")
    
    # Sample size
    I = len(Z_resid)
    
    # Compute pseudo-outcomes for treatment
    pseudo = hat_mu + (A / pi) * (M - hat_mu)
    
    # Compute numerator and denominator for IV estimator
    hat_theta_num = np.mean(Z_resid * Y) 
    hat_theta_denom = np.mean(Z_resid * pseudo)
    
    if abs(hat_theta_denom) < 1e-10:
        warnings.warn("Weak instruments: denominator near zero")
    
    # IV estimate  
    hat_theta = hat_theta_num / hat_theta_denom
    
    # Compute asymptotic variance using influence function
    psi = (1 / hat_theta_denom) * Z_resid * (Y - pseudo * hat_theta)
    avar = np.mean(psi**2)
    
    # Standard error
    se = np.sqrt(avar / I)
    
    # Confidence interval
    ci_width = stats.norm.ppf((1 + level) / 2) * se
    
    return {
        'est': hat_theta,
        'se': se,
        'ci_lower': hat_theta - ci_width,
        'ci_upper': hat_theta + ci_width
    }


def mars_did_estimator(
    M: np.ndarray,
    C: np.ndarray, 
    G: np.ndarray,
    A: np.ndarray,
    muG: np.ndarray,
    muC: np.ndarray,
    pi: np.ndarray,
    probG: float,
    probC: float,
    level: float = 0.95
) -> Dict[str, float]:
    """
    MARS Difference-in-Differences Estimator
    
    Computes MARS DiD estimates following Callaway and Sant'Anna (2021) 
    methodology with pseudo-outcomes for staggered treatment adoption.
    
    Parameters
    ----------
    M : np.ndarray
        Ground truth outcome values
    C : np.ndarray
        Indicators for never-treated units
    G : np.ndarray
        Staggered treatment indicators  
    A : np.ndarray
        Annotation indicators
    muG : np.ndarray
        ML predictions for treated units
    muC : np.ndarray
        ML predictions for control units
    pi : np.ndarray
        Annotation probabilities
    probG : float
        Probability of being in treated cohort
    probC : float
        Probability of being never-treated
    level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, float]
        Dictionary with DiD estimate, standard error, and confidence interval
        
    Examples
    --------
    >>> result = mars_did_estimator(
    ...     M=ground_truth_outcomes,
    ...     C=never_treated_indicators,
    ...     G=treatment_cohort_indicators,
    ...     A=annotation_indicators,
    ...     muG=ml_predictions_treated,
    ...     muC=ml_predictions_control,
    ...     pi=annotation_probs,
    ...     probG=0.3,
    ...     probC=0.7
    ... )
    """
    # Input validation
    arrays = [M, C, G, A, muG, muC, pi]
    M, C, G, A, muG, muC, pi = map(np.asarray, arrays)
    
    if not all(len(arr) == len(M) for arr in arrays):
        raise ValueError("All input arrays must have the same length")
        
    if probG <= 0 or probC <= 0:
        raise ValueError("Probabilities must be positive")
        
    if np.any(pi <= 0):
        raise ValueError("All annotation probabilities must be positive")
    
    # Sample size
    I = len(M)
    
    # Compute pseudo-outcomes for both groups
    pseudoC = muC + (A / pi) * (M - muC)
    pseudoG = muG + (A / pi) * (M - muG)
    
    # Compute group-specific estimates
    hat_theta_G = np.mean((G / probG) * pseudoG)
    hat_theta_C = np.mean((C / probC) * pseudoC)
    hat_theta = hat_theta_G - hat_theta_C
    
    # Compute asymptotic variance using influence function
    psi = (G / probG) * pseudoG - (C / probC) * pseudoC - hat_theta
    avar = np.mean(psi**2)
    
    # Standard error
    se = np.sqrt(avar / I)
    
    # Confidence interval
    ci_width = stats.norm.ppf((1 + level) / 2) * se
    
    return {
        'est': hat_theta,
        'se': se,
        'ci_lower': hat_theta - ci_width,
        'ci_upper': hat_theta + ci_width
    }


def mars_lrdd_estimator(
    M: np.ndarray,
    R: np.ndarray,
    D: np.ndarray, 
    A: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
    pi: np.ndarray,
    B: List[float],
    prob1: float,
    prob0: float, 
    level: float = 0.95
) -> Dict[str, float]:
    """
    MARS Local Regression Discontinuity Estimator
    
    Computes MARS local RDD estimates and confidence intervals for treatment
    effects at the discontinuity threshold using pseudo-outcomes.
    
    Parameters
    ----------
    M : np.ndarray
        Ground truth outcome values
    R : np.ndarray
        Running variable values
    D : np.ndarray
        Treatment indicators
    A : np.ndarray
        Annotation indicators
    mu1 : np.ndarray
        ML predictions for treated units
    mu0 : np.ndarray
        ML predictions for control units
    pi : np.ndarray
        Annotation probabilities
    B : List[float]
        Window bounds [lower_bound, upper_bound] around threshold
    prob1 : float
        Probability of being treated within window
    prob0 : float
        Probability of being control within window
    level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, float]
        Dictionary with RDD estimate, standard error, and confidence interval
        
    Examples
    --------
    >>> result = mars_lrdd_estimator(
    ...     M=ground_truth_outcomes,
    ...     R=running_variable,
    ...     D=treatment_indicators,
    ...     A=annotation_indicators,
    ...     mu1=ml_predictions_treated,
    ...     mu0=ml_predictions_control,
    ...     pi=annotation_probs,
    ...     B=[-2.0, 2.0],
    ...     prob1=0.5,
    ...     prob0=0.5
    ... )
    """
    # Input validation
    arrays = [M, R, D, A, mu1, mu0, pi]
    M, R, D, A, mu1, mu0, pi = map(np.asarray, arrays)
    
    if not all(len(arr) == len(M) for arr in arrays):
        raise ValueError("All input arrays must have the same length")
        
    if len(B) != 2:
        raise ValueError("B must contain exactly two bounds [lower, upper]")
        
    if prob1 <= 0 or prob0 <= 0:
        raise ValueError("Probabilities must be positive")
        
    if np.any(pi <= 0):
        raise ValueError("All annotation probabilities must be positive")
    
    # Sample size
    I = len(M)
    
    # Indicator for running variable within window B
    R_in_B = (R >= B[0]) & (R <= B[1])
    D_is_1 = (D == 1).astype(float)
    D_is_0 = (D == 0).astype(float)
    
    # Compute pseudo-outcomes
    pseudo1 = mu1 + (A / pi) * (M - mu1)
    pseudo0 = mu0 + (A / pi) * (M - mu0)
    
    # Compute treatment and control estimates within window
    hat_theta_1 = np.mean(((R_in_B * D_is_1) / prob1) * pseudo1)
    hat_theta_0 = np.mean(((R_in_B * D_is_0) / prob0) * pseudo0)
    hat_theta = hat_theta_1 - hat_theta_0
    
    # Compute asymptotic variance using influence function
    psi = (((R_in_B * D_is_1) / prob1) * pseudo1 - 
           ((R_in_B * D_is_0) / prob0) * pseudo0 - hat_theta)
    avar = np.mean(psi**2)
    
    # Standard error
    se = np.sqrt(avar / I)
    
    # Confidence interval
    ci_width = stats.norm.ppf((1 + level) / 2) * se
    
    return {
        'est': hat_theta,
        'se': se,
        'ci_lower': hat_theta - ci_width,
        'ci_upper': hat_theta + ci_width
    }