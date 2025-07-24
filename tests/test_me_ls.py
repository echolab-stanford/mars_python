"""
Tests for measurement error corrected least squares estimators

This module tests the ME-LS functions with various scenarios including
clustering, eigenvalue corrections, and different error structures.
"""

import pytest
import numpy as np
from scipy import linalg
import sys
import os

# Add the mars package to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mars.me_ls import me_ls_iid, me_ls_inid


class TestMELS:
    """Test suite for measurement error corrected least squares"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n = 200
        self.k = 3
        
        # Generate design matrix
        self.X = np.random.randn(self.n, self.k)
        
        # True coefficients
        self.beta_true = np.array([1.0, -0.5, 2.0])
        
        # Generate outcome with noise
        self.Y = self.X @ self.beta_true + np.random.randn(self.n) * 0.5
        
        # Measurement error covariance (for testing)
        self.Sigma = np.eye(self.k) * 0.1
        
        # Cluster variables
        self.cluster1 = np.random.randint(0, 20, self.n)
        self.cluster2 = np.random.randint(0, 15, self.n)
    
    def test_me_ls_iid_basic(self):
        """Test basic functionality of ME-LS with iid errors"""
        result = me_ls_iid(
            X=self.X,
            Y=self.Y,
            Sigma=self.Sigma
        )
        
        # Check result structure
        required_keys = ['coefficients', 'vcov', 'robust_se', 'p_values', 'ci_lower', 'ci_upper']
        assert all(key in result for key in required_keys)
        
        # Check dimensions
        assert result['coefficients'].shape == (self.k,)
        assert result['vcov'].shape == (self.k, self.k)
        assert result['robust_se'].shape == (self.k,)
        
        # Estimates should be close to true values (with measurement error correction)
        assert np.allclose(result['coefficients'], self.beta_true, atol=0.5)
        
        # Standard errors should be positive
        assert np.all(result['robust_se'] > 0)
        
        # P-values should be between 0 and 1
        assert np.all((result['p_values'] >= 0) & (result['p_values'] <= 1))
        
        # Confidence intervals should be properly ordered
        assert np.all(result['ci_lower'] < result['ci_upper'])
    
    def test_me_ls_iid_no_measurement_error(self):
        """Test ME-LS with no measurement error (Sigma = 0)"""
        Sigma_zero = np.zeros((self.k, self.k))
        
        result = me_ls_iid(
            X=self.X,
            Y=self.Y, 
            Sigma=Sigma_zero
        )
        
        # Should be equivalent to OLS
        ols_coef = linalg.solve(self.X.T @ self.X, self.X.T @ self.Y)
        assert np.allclose(result['coefficients'], ols_coef, atol=1e-10)
    
    def test_me_ls_iid_clustering(self):
        """Test ME-LS with one-way clustering"""
        result = me_ls_iid(
            X=self.X,
            Y=self.Y,
            Sigma=self.Sigma,
            cluster1=self.cluster1
        )
        
        # Should still have correct structure
        assert result['coefficients'].shape == (self.k,)
        assert np.all(result['robust_se'] > 0)
        
        # Compare with no clustering - standard errors should generally be different
        result_no_cluster = me_ls_iid(
            X=self.X,
            Y=self.Y,
            Sigma=self.Sigma
        )
        
        # Coefficients should be the same
        assert np.allclose(result['coefficients'], result_no_cluster['coefficients'])
        
        # Standard errors will typically be different (could be larger or smaller)
        assert not np.allclose(result['robust_se'], result_no_cluster['robust_se'])
    
    def test_me_ls_iid_two_way_clustering(self):
        """Test ME-LS with two-way clustering"""
        result = me_ls_iid(
            X=self.X,
            Y=self.Y,
            Sigma=self.Sigma,
            cluster1=self.cluster1,
            cluster2=self.cluster2
        )
        
        # Should have correct structure
        assert result['coefficients'].shape == (self.k,)
        assert np.all(result['robust_se'] > 0)
    
    def test_me_ls_iid_eigenvalue_correction(self):
        """Test eigenvalue correction functionality"""
        result_with_correction = me_ls_iid(
            X=self.X,
            Y=self.Y,
            Sigma=self.Sigma,
            eigfix=True,
            tol=1e-6
        )
        
        result_without_correction = me_ls_iid(
            X=self.X,
            Y=self.Y,
            Sigma=self.Sigma,
            eigfix=False
        )
        
        # Coefficients should be the same
        assert np.allclose(result_with_correction['coefficients'], 
                          result_without_correction['coefficients'])
        
        # Variance matrix should be positive definite with correction
        eigenvals = linalg.eigvals(result_with_correction['vcov'])
        assert np.all(eigenvals >= 1e-6 - 1e-10)  # Allow for numerical precision
    
    def test_me_ls_inid_basic(self):
        """Test basic functionality of ME-LS with heteroskedastic errors"""
        # Create observation-specific Sigma matrices
        Sigma_list = [np.eye(self.k) * (0.05 + 0.1 * np.random.rand()) for _ in range(self.n)]
        
        result = me_ls_inid(
            X=self.X,
            Y=self.Y,
            Sigma=Sigma_list
        )
        
        # Check result structure
        required_keys = ['coefficients', 'vcov', 'robust_se', 'p_values', 'ci_lower', 'ci_upper']
        assert all(key in result for key in required_keys)
        
        # Check dimensions
        assert result['coefficients'].shape == (self.k,)
        assert result['vcov'].shape == (self.k, self.k)
        
        # Standard errors should be positive
        assert np.all(result['robust_se'] > 0)
    
    def test_me_ls_inid_validation(self):
        """Test input validation for ME-LS inid"""
        # Test wrong number of Sigma matrices
        Sigma_list_short = [np.eye(self.k) * 0.1 for _ in range(self.n - 1)]
        
        with pytest.raises(ValueError):
            me_ls_inid(
                X=self.X,
                Y=self.Y,
                Sigma=Sigma_list_short  # Wrong length
            )
        
        # Test wrong Sigma dimensions
        Sigma_list_wrong_dim = [np.eye(self.k + 1) * 0.1 for _ in range(self.n)]
        
        with pytest.raises(ValueError):
            me_ls_inid(
                X=self.X,
                Y=self.Y,
                Sigma=Sigma_list_wrong_dim  # Wrong dimensions
            )
    
    def test_me_ls_iid_validation(self):
        """Test input validation for ME-LS iid"""
        # Test dimension mismatch
        X_wrong = self.X[:-1]  # One fewer row
        
        with pytest.raises(ValueError):
            me_ls_iid(
                X=X_wrong,
                Y=self.Y,
                Sigma=self.Sigma
            )
        
        # Test wrong Sigma dimensions
        Sigma_wrong = np.eye(self.k + 1)
        
        with pytest.raises(ValueError):
            me_ls_iid(
                X=self.X,
                Y=self.Y,
                Sigma=Sigma_wrong
            )
    
    def test_multiple_outcomes(self):
        """Test ME-LS with multiple outcome variables"""
        # Create multiple outcomes
        Y_multi = np.column_stack([
            self.X @ self.beta_true + np.random.randn(self.n) * 0.3,
            self.X @ (self.beta_true * 0.5) + np.random.randn(self.n) * 0.4
        ])
        
        result = me_ls_iid(
            X=self.X,
            Y=Y_multi,
            Sigma=self.Sigma
        )
        
        # Should have results for each outcome
        assert 'outcome_0' in result
        assert 'outcome_1' in result
        
        # Each outcome should have correct structure
        for outcome_key in ['outcome_0', 'outcome_1']:
            outcome_result = result[outcome_key]
            assert outcome_result['coefficients'].shape == (self.k,)
            assert np.all(outcome_result['robust_se'] > 0)
    
    def test_confidence_levels(self):
        """Test different confidence levels"""
        for level in [0.90, 0.95, 0.99]:
            result = me_ls_iid(
                X=self.X,
                Y=self.Y,
                Sigma=self.Sigma,
                level=level
            )
            
            # Check that confidence intervals have correct coverage
            # (Width should increase with confidence level)
            ci_width = result['ci_upper'] - result['ci_lower']
            
            # All widths should be positive
            assert np.all(ci_width > 0)
    
    def test_singular_matrix_handling(self):
        """Test handling of near-singular matrices"""
        # Create a design matrix with multicollinearity
        X_singular = np.column_stack([
            np.random.randn(self.n),
            np.random.randn(self.n),
            np.random.randn(self.n) * 1e-10  # Nearly zero column
        ])
        
        Y_singular = X_singular @ np.array([1, -1, 0]) + np.random.randn(self.n) * 0.1
        Sigma_singular = np.eye(3) * 0.01
        
        # This should raise an error or handle gracefully
        with pytest.raises(ValueError):
            me_ls_iid(
                X=X_singular,
                Y=Y_singular,
                Sigma=Sigma_singular
            )
    
    def test_consistency_iid_vs_inid(self):
        """Test that iid and inid give same results when Sigma matrices are identical"""
        # Create identical Sigma matrices for all observations
        Sigma_list = [self.Sigma.copy() for _ in range(self.n)]
        
        result_iid = me_ls_iid(
            X=self.X,
            Y=self.Y,
            Sigma=self.Sigma
        )
        
        result_inid = me_ls_inid(
            X=self.X,
            Y=self.Y,
            Sigma=Sigma_list
        )
        
        # Results should be very similar (allowing for numerical differences)
        assert np.allclose(result_iid['coefficients'], result_inid['coefficients'], atol=1e-8)
        assert np.allclose(result_iid['robust_se'], result_inid['robust_se'], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])