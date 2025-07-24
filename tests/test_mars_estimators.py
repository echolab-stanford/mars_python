"""
Tests for MARS estimators

This module provides comprehensive tests for all MARS statistical estimators,
including validation against known results and consistency checks.
"""

import pytest
import numpy as np
from scipy import stats
import sys
import os

# Add the mars package to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mars.mars_estimators import (
    mars_mean_estimator,
    mars_linreg_estimator,
    mars_liv_estimator, 
    mars_did_estimator,
    mars_lrdd_estimator
)
from mars.utils import simulate_mars_data


class TestMarsEstimators:
    """Test suite for MARS estimators"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.data = simulate_mars_data(n=1000, annotation_rate=0.3)
    
    def test_mars_mean_estimator_basic(self):
        """Test basic functionality of MARS mean estimator"""
        result = mars_mean_estimator(
            M=self.data['M'],
            A=self.data['A'],
            hat_mu=self.data['hat_mu'],
            pi=self.data['pi']
        )
        
        # Check result structure
        required_keys = ['est', 'se', 'ci_lower', 'ci_upper']
        assert all(key in result for key in required_keys)
        
        # Check that estimate is reasonable
        true_mean = np.mean(self.data['Y_true'])
        assert abs(result['est'] - true_mean) < 0.5  # Should be close
        
        # Check that standard error is positive
        assert result['se'] > 0
        
        # Check confidence interval
        assert result['ci_lower'] < result['est'] < result['ci_upper']
    
    def test_mars_mean_estimator_perfect_ml(self):
        """Test MARS mean estimator with perfect ML predictions"""
        # When ML predictions are perfect, estimate should equal true mean
        result = mars_mean_estimator(
            M=self.data['Y_true'],  # Perfect ground truth where annotated
            A=self.data['A'],
            hat_mu=self.data['Y_true'],  # Perfect ML predictions
            pi=self.data['pi']
        )
        
        true_mean = np.mean(self.data['Y_true'])
        assert abs(result['est'] - true_mean) < 1e-10
    
    def test_mars_mean_estimator_input_validation(self):
        """Test input validation for MARS mean estimator"""
        # Test mismatched lengths
        with pytest.raises(ValueError):
            mars_mean_estimator(
                M=self.data['M'][:-1],  # Shorter array
                A=self.data['A'], 
                hat_mu=self.data['hat_mu'],
                pi=self.data['pi']
            )
        
        # Test negative annotation probabilities
        pi_bad = self.data['pi'].copy()
        pi_bad[0] = -0.1
        with pytest.raises(ValueError):
            mars_mean_estimator(
                M=self.data['M'],
                A=self.data['A'],
                hat_mu=self.data['hat_mu'],
                pi=pi_bad
            )
    
    def test_mars_linreg_estimator_basic(self):
        """Test basic functionality of MARS linear regression estimator"""
        # Create residualized covariate (mean-zero)
        X = np.random.randn(len(self.data['M']))
        C_resid = X - np.mean(X)  # Make mean-zero
        
        result = mars_linreg_estimator(
            C_resid=C_resid,
            A=self.data['A'],
            M=self.data['M'],
            hat_mu=self.data['hat_mu'],
            pi=self.data['pi']
        )
        
        # Check result structure
        required_keys = ['est', 'se', 'ci_lower', 'ci_upper']
        assert all(key in result for key in required_keys)
        
        # Check standard error is positive
        assert result['se'] > 0
        
        # Check confidence interval
        assert result['ci_lower'] < result['ci_upper']
    
    def test_mars_linreg_estimator_validation(self):
        """Test input validation for linear regression estimator"""
        # Test non-mean-zero covariate
        C_not_centered = np.random.randn(len(self.data['M'])) + 5  # Has non-zero mean
        
        with pytest.raises(ValueError):
            mars_linreg_estimator(
                C_resid=C_not_centered,
                A=self.data['A'],
                M=self.data['M'],
                hat_mu=self.data['hat_mu'],
                pi=self.data['pi']
            )
    
    def test_mars_liv_estimator_basic(self):
        """Test basic functionality of MARS IV estimator"""
        n = len(self.data['M'])
        Z_resid = np.random.randn(n) - np.mean(np.random.randn(n))  # Residualized instrument
        Y = np.random.randn(n)  # Outcome
        
        result = mars_liv_estimator(
            Z_resid=Z_resid,
            Y=Y,
            M=self.data['M'],
            A=self.data['A'],
            hat_mu=self.data['hat_mu'],
            pi=self.data['pi']
        )
        
        # Check result structure
        required_keys = ['est', 'se', 'ci_lower', 'ci_upper']
        assert all(key in result for key in required_keys)
        
        # Check standard error is positive  
        assert result['se'] > 0
    
    def test_mars_did_estimator_basic(self):
        """Test basic functionality of MARS DiD estimator"""
        n = len(self.data['M'])
        
        # Create treatment groups
        C = np.random.binomial(1, 0.7, n)  # Never-treated
        G = 1 - C  # Treated group
        
        # ML predictions for each group
        muG = self.data['hat_mu'] + np.random.randn(n) * 0.1
        muC = self.data['hat_mu'] + np.random.randn(n) * 0.1
        
        result = mars_did_estimator(
            M=self.data['M'],
            C=C,
            G=G,
            A=self.data['A'],
            muG=muG,
            muC=muC,
            pi=self.data['pi'],
            probG=0.3,
            probC=0.7
        )
        
        # Check result structure
        required_keys = ['est', 'se', 'ci_lower', 'ci_upper']
        assert all(key in result for key in required_keys)
        
        # Check standard error is positive
        assert result['se'] > 0
    
    def test_mars_did_estimator_validation(self):
        """Test input validation for DiD estimator"""
        n = len(self.data['M'])
        C = np.random.binomial(1, 0.7, n)
        G = 1 - C
        muG = self.data['hat_mu']
        muC = self.data['hat_mu']
        
        # Test negative probabilities
        with pytest.raises(ValueError):
            mars_did_estimator(
                M=self.data['M'],
                C=C, G=G, A=self.data['A'],
                muG=muG, muC=muC, pi=self.data['pi'],
                probG=-0.1,  # Negative probability
                probC=0.7
            )
    
    def test_mars_lrdd_estimator_basic(self):
        """Test basic functionality of MARS RDD estimator"""
        n = len(self.data['M'])
        
        # Create RDD setup
        R = np.random.uniform(-5, 5, n)  # Running variable
        D = (R >= 0).astype(int)  # Treatment indicator
        
        # ML predictions
        mu1 = self.data['hat_mu'] + np.random.randn(n) * 0.1
        mu0 = self.data['hat_mu'] + np.random.randn(n) * 0.1
        
        result = mars_lrdd_estimator(
            M=self.data['M'],
            R=R,
            D=D,
            A=self.data['A'],
            mu1=mu1,
            mu0=mu0,
            pi=self.data['pi'],
            B=[-2.0, 2.0],  # Window around threshold
            prob1=0.5,
            prob0=0.5
        )
        
        # Check result structure  
        required_keys = ['est', 'se', 'ci_lower', 'ci_upper']
        assert all(key in result for key in required_keys)
        
        # Check standard error is positive
        assert result['se'] > 0
    
    def test_mars_lrdd_estimator_validation(self):
        """Test input validation for RDD estimator"""
        n = len(self.data['M'])
        R = np.random.uniform(-5, 5, n)
        D = (R >= 0).astype(int)
        mu1 = self.data['hat_mu']
        mu0 = self.data['hat_mu']
        
        # Test invalid window bounds
        with pytest.raises(ValueError):
            mars_lrdd_estimator(
                M=self.data['M'], R=R, D=D, A=self.data['A'],
                mu1=mu1, mu0=mu0, pi=self.data['pi'],
                B=[-2.0],  # Only one bound
                prob1=0.5, prob0=0.5
            )
    
    def test_confidence_levels(self):
        """Test different confidence levels work correctly"""
        for level in [0.90, 0.95, 0.99]:
            result = mars_mean_estimator(
                M=self.data['M'],
                A=self.data['A'],
                hat_mu=self.data['hat_mu'], 
                pi=self.data['pi'],
                level=level
            )
            
            # Width should increase with confidence level
            width = result['ci_upper'] - result['ci_lower']
            expected_width = 2 * stats.norm.ppf((1 + level) / 2) * result['se']
            assert abs(width - expected_width) < 1e-10
    
    def test_scaling_parameter(self):
        """Test scaling parameter in mean estimator"""
        scale = 2.0
        result_scaled = mars_mean_estimator(
            M=self.data['M'],
            A=self.data['A'],
            hat_mu=self.data['hat_mu'],
            pi=self.data['pi'],
            scale=scale
        )
        
        result_unscaled = mars_mean_estimator(
            M=self.data['M'],
            A=self.data['A'],
            hat_mu=self.data['hat_mu'],
            pi=self.data['pi'],
            scale=1.0
        )
        
        # Estimate should be scaled
        assert abs(result_scaled['est'] - scale * result_unscaled['est']) < 1e-10
        
        # Standard error should also be scaled
        assert abs(result_scaled['se'] - scale * result_unscaled['se']) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])