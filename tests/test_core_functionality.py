"""
Basic tests for LQG Polymer Field Generator core functionality.

This test module validates the essential mathematical operations and
ensures the implementation maintains physical consistency.

Author: LQG-FTL Research Team
Date: July 2025
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from core.polymer_quantization import PolymerQuantization, PolymerFieldGenerator
    from core.field_operators import PolymerFieldOperator, QuantumGeometricFieldAlgebra
    from optimization.quantum_inequality import QuantumInequalityBounds, NegativeEnergyGenerator
    from field_generation.spatial_configuration import SpatialFieldProfile
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

class TestPolymerQuantization:
    """Test core polymer quantization functionality."""
    
    def test_sinc_enhancement_factor(self):
        """Test sinc(πμ) calculation."""
        polymer = PolymerQuantization(mu=0.7)
        sinc_factor = polymer.sinc_enhancement_factor()
        
        # Verify sinc function properties
        assert 0 < sinc_factor < 1, "Sinc factor should be between 0 and 1 for μ > 0"
        
        # Test edge case μ = 0
        polymer_zero = PolymerQuantization(mu=0.0)
        assert polymer_zero.sinc_enhancement_factor() == 1.0, "sinc(0) should equal 1"
        
        # Test specific value
        expected_sinc = np.sin(np.pi * 0.7) / (np.pi * 0.7)
        assert abs(sinc_factor - expected_sinc) < 1e-10, "Sinc calculation error"
    
    def test_polymer_momentum_substitution(self):
        """Test polymer momentum substitution."""
        polymer = PolymerQuantization(mu=0.7)
        classical_p = 1.0
        
        polymer_p = polymer.polymer_momentum_substitution(classical_p)
        
        # Verify momentum is bounded
        max_momentum = polymer.hbar / polymer.mu
        assert abs(polymer_p) <= max_momentum, "Polymer momentum should be bounded"
        
        # Test limiting behavior
        assert isinstance(polymer_p, (int, float, complex)), "Should return numeric value"
    
    def test_enhancement_magnitude(self):
        """Test enhancement magnitude calculation."""
        polymer = PolymerQuantization(mu=0.7)
        enhancement = polymer.enhancement_magnitude()
        
        assert enhancement > 0, "Enhancement should be positive"
        assert enhancement > 1e9, "Enhancement should be significant (>1 billion)"
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameter
        polymer_valid = PolymerQuantization(mu=0.5)
        assert polymer_valid.mu == 0.5
        
        # Invalid parameter
        with pytest.raises(ValueError):
            PolymerQuantization(mu=-0.1)

class TestQuantumFieldOperators:
    """Test quantum field operator implementations."""
    
    def test_field_operator_matrix_elements(self):
        """Test field operator matrix elements."""
        field_op = PolymerFieldOperator(mu=0.7)
        phi_test = 1.0
        
        # Diagonal element
        diagonal_element = field_op.field_operator_matrix_element(0, 0, phi_test)
        expected = phi_test * field_op.sinc_factor()
        assert abs(diagonal_element - expected) < 1e-10, "Diagonal element calculation error"
        
        # Off-diagonal element
        off_diagonal = field_op.field_operator_matrix_element(0, 1, phi_test)
        assert off_diagonal == 0.0, "Off-diagonal elements should be zero"
    
    def test_commutator_relations(self):
        """Test commutator relations."""
        field_op = PolymerFieldOperator(mu=0.7)
        
        # Test [Φ, Π] commutator
        commutator = field_op.commutator_matrix_element(0, 0)
        expected = 1j * field_op.hbar * field_op.sinc_factor()
        
        assert abs(commutator - expected) < 1e-10, "Commutator calculation error"
        assert commutator.imag != 0, "Commutator should be imaginary"
    
    def test_uncertainty_relation(self):
        """Test quantum uncertainty relation."""
        qg_algebra = QuantumGeometricFieldAlgebra(mu=0.7)
        
        # Normalized state
        state_amps = np.array([0.8, 0.6])
        state_amps = state_amps / np.linalg.norm(state_amps)
        phi_vals = np.array([1.0, -0.5])
        
        delta_phi, delta_pi, min_uncertainty = qg_algebra.uncertainty_relation(state_amps, phi_vals)
        
        # Verify uncertainty relation
        assert delta_phi >= 0, "Field uncertainty should be non-negative"
        assert delta_pi >= 0, "Momentum uncertainty should be non-negative"
        assert min_uncertainty > 0, "Minimum uncertainty should be positive"

class TestQuantumInequality:
    """Test quantum inequality enhancements."""
    
    def test_ford_roman_bounds(self):
        """Test Ford-Roman bound calculations."""
        qi_bounds = QuantumInequalityBounds(mu=0.7, tau=1.0)
        
        classical_bound = qi_bounds.classical_ford_roman_bound()
        enhanced_bound = qi_bounds.enhanced_ford_roman_bound()
        
        # Bounds should be negative
        assert classical_bound < 0, "Classical bound should be negative"
        assert enhanced_bound < 0, "Enhanced bound should be negative"
        
        # Enhanced bound should allow stronger violations
        assert abs(enhanced_bound) >= abs(classical_bound), "Enhanced bound should be stronger"
    
    def test_negative_energy_generation(self):
        """Test negative energy generation."""
        neg_energy = NegativeEnergyGenerator(mu=0.7, tau=1.0)
        
        t = np.linspace(-2, 2, 100)
        rho_eff = neg_energy.energy_density_profile(t, amplitude=1.0)
        
        # Should contain negative energy regions
        assert np.any(rho_eff < 0), "Should generate negative energy regions"
        
        # Energy density should be real
        assert np.all(np.isreal(rho_eff)), "Energy density should be real"
    
    def test_quantum_inequality_validation(self):
        """Test quantum inequality validation."""
        neg_energy = NegativeEnergyGenerator(mu=0.7, tau=1.0)
        
        t = np.linspace(-5, 5, 200)
        rho_eff = neg_energy.energy_density_profile(t, amplitude=0.5)  # Small amplitude
        f_sampling = neg_energy.qi_bounds.optimal_sampling_function(t)
        
        validation = neg_energy.validate_quantum_inequality(t, rho_eff, f_sampling)
        
        assert 'is_valid' in validation, "Validation should include validity check"
        assert 'integral_value' in validation, "Should calculate integral value"
        assert validation['violation_strength'] > 1.0, "Should show enhancement"

class TestSpatialConfiguration:
    """Test spatial field configuration."""
    
    def test_shape_functions(self):
        """Test various shape functions."""
        spatial_profile = SpatialFieldProfile(mu=0.7, R_s=1.0)
        
        r = np.linspace(0, 5, 50)
        
        # Test Gaussian shape
        gaussian = spatial_profile.gaussian_shape_function(r)
        assert gaussian[0] == 1.0, "Gaussian should equal 1 at r=0"
        assert np.all(gaussian > 0), "Gaussian should be positive"
        assert np.all(gaussian <= 1.0), "Gaussian should be bounded by 1"
        
        # Test Lorentzian shape
        lorentzian = spatial_profile.lorentzian_shape_function(r)
        assert lorentzian[0] == 1.0, "Lorentzian should equal 1 at r=0"
        assert np.all(lorentzian > 0), "Lorentzian should be positive"
    
    def test_enhancement_field_generation(self):
        """Test enhanced field generation."""
        spatial_profile = SpatialFieldProfile(mu=0.7, R_s=1.0)
        
        x = np.linspace(-3, 3, 50)
        field = spatial_profile.enhancement_field_1d(x, phi_0=1.0, shape_type='gaussian')
        
        assert len(field) == len(x), "Field should have same length as coordinates"
        assert np.all(np.isfinite(field)), "Field should be finite everywhere"
        
        # Maximum should be at center (for symmetric profiles)
        center_idx = len(x) // 2
        assert field[center_idx] == np.max(field), "Maximum should be at center"
    
    def test_3d_field_generation(self):
        """Test 3D field generation."""
        spatial_profile = SpatialFieldProfile(mu=0.7, R_s=1.0)
        
        # Small 3D grid for testing
        n = 10
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        z = np.linspace(-1, 1, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        field_3d = spatial_profile.enhancement_field_3d(X, Y, Z, shape_type='gaussian')
        
        assert field_3d.shape == (n, n, n), "3D field should have correct shape"
        assert np.all(np.isfinite(field_3d)), "3D field should be finite"

class TestIntegration:
    """Test integration between components."""
    
    def test_field_generator_integration(self):
        """Test integration of field generator components."""
        generator = PolymerFieldGenerator(mu_optimal=0.7)
        
        # Test field generation
        spatial_coords = np.linspace(-5, 5, 100)
        field = generator.generate_sinc_enhancement_field(spatial_coords=spatial_coords)
        
        assert len(field) == len(spatial_coords), "Field length should match coordinates"
        assert np.all(np.isfinite(field)), "Generated field should be finite"
        
        # Test statistics calculation
        stats = generator.field_enhancement_statistics()
        
        required_keys = ['sinc_enhancement', 'total_enhancement', 'backreaction_coupling']
        for key in required_keys:
            assert key in stats, f"Statistics should include {key}"
            assert np.isfinite(stats[key]), f"{key} should be finite"
    
    def test_parameter_consistency(self):
        """Test parameter consistency across components."""
        mu_test = 0.8
        
        # Initialize multiple components with same μ
        polymer = PolymerQuantization(mu=mu_test)
        field_op = PolymerFieldOperator(mu=mu_test)
        qi_bounds = QuantumInequalityBounds(mu=mu_test)
        spatial = SpatialFieldProfile(mu=mu_test)
        
        # All should give same sinc factor
        sinc_polymer = polymer.sinc_enhancement_factor()
        sinc_field = field_op.sinc_factor()
        sinc_qi = qi_bounds.sinc_factor()
        sinc_spatial = spatial.sinc_factor()
        
        tolerance = 1e-10
        assert abs(sinc_polymer - sinc_field) < tolerance, "Sinc factors should be consistent"
        assert abs(sinc_polymer - sinc_qi) < tolerance, "Sinc factors should be consistent"
        assert abs(sinc_polymer - sinc_spatial) < tolerance, "Sinc factors should be consistent"

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
