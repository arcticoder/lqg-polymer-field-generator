"""
Robust Optimization Framework for LQG Polymer Field Generator

This module provides enhanced optimization with improved convergence,
parameter validation, and uncertainty handling to resolve critical UQ concerns.

Author: LQG-FTL Research Team
Date: July 2025
"""

import numpy as np
import scipy.constants as const
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution, OptimizeResult
import warnings
import logging

logger = logging.getLogger(__name__)

class RobustParameterValidator:
    """
    Robust parameter validation with safe operating ranges.
    
    Addresses UQ concerns:
    - Parameter sensitivity issues
    - Numerical instability from extreme values
    - Physical constraint violations
    """
    
    def __init__(self):
        """Initialize parameter validator with safe ranges."""
        # Safe operating ranges based on UQ analysis
        self.safe_ranges = {
            'mu': (1e-6, 2.0),          # Polymer parameter
            'tau': (1e-3, 100.0),       # Timescale parameter
            'amplitude': (1e-6, 10.0),   # Field amplitude
            'n_points': (10, 10000),     # Array sizes
            'tolerance': (1e-12, 1e-3)   # Numerical tolerances
        }
        
        # Recommended nominal values for stability
        self.nominal_values = {
            'mu': 0.7,
            'tau': 1.0,
            'amplitude': 1.0
        }
    
    def validate_mu(self, mu: float, strict: bool = True) -> Tuple[float, List[str]]:
        """
        Validate and potentially correct polymer parameter μ.
        
        Parameters:
        -----------
        mu : float
            Input polymer parameter
        strict : bool
            Whether to enforce strict bounds or apply corrections
            
        Returns:
        --------
        Tuple[float, List[str]]
            (validated_mu, warnings)
        """
        warnings_list = []
        validated_mu = mu
        
        mu_min, mu_max = self.safe_ranges['mu']
        
        if not np.isfinite(mu):
            validated_mu = self.nominal_values['mu']
            warnings_list.append(f"Non-finite μ={mu} replaced with nominal μ={validated_mu}")
        elif mu <= 0:
            if strict:
                raise ValueError(f"Polymer parameter μ must be positive, got μ={mu}")
            else:
                validated_mu = mu_min
                warnings_list.append(f"Non-positive μ={mu} corrected to μ={validated_mu}")
        elif mu < mu_min:
            if strict:
                raise ValueError(f"μ={mu} below safe minimum {mu_min}")
            else:
                validated_mu = mu_min
                warnings_list.append(f"μ={mu} below safe range, corrected to μ={validated_mu}")
        elif mu > mu_max:
            if strict:
                warnings_list.append(f"WARNING: μ={mu} above recommended maximum {mu_max}")
            else:
                validated_mu = mu_max
                warnings_list.append(f"μ={mu} above safe range, corrected to μ={validated_mu}")
        
        return validated_mu, warnings_list
    
    def validate_tau(self, tau: float, strict: bool = True) -> Tuple[float, List[str]]:
        """
        Validate and potentially correct timescale parameter τ.
        
        Parameters:
        -----------
        tau : float
            Input timescale parameter
        strict : bool
            Whether to enforce strict bounds or apply corrections
            
        Returns:
        --------
        Tuple[float, List[str]]
            (validated_tau, warnings)
        """
        warnings_list = []
        validated_tau = tau
        
        tau_min, tau_max = self.safe_ranges['tau']
        
        if not np.isfinite(tau):
            validated_tau = self.nominal_values['tau']
            warnings_list.append(f"Non-finite τ={tau} replaced with nominal τ={validated_tau}")
        elif tau <= 0:
            if strict:
                raise ValueError(f"Timescale parameter τ must be positive, got τ={tau}")
            else:
                validated_tau = tau_min
                warnings_list.append(f"Non-positive τ={tau} corrected to τ={validated_tau}")
        elif tau < tau_min:
            if strict:
                raise ValueError(f"τ={tau} below safe minimum {tau_min}")
            else:
                validated_tau = tau_min
                warnings_list.append(f"τ={tau} below safe range, corrected to τ={validated_tau}")
        elif tau > tau_max:
            if strict:
                warnings_list.append(f"WARNING: τ={tau} above recommended maximum {tau_max}")
            else:
                validated_tau = tau_max
                warnings_list.append(f"τ={tau} above safe range, corrected to τ={validated_tau}")
        
        return validated_tau, warnings_list
    
    def validate_all_parameters(self, params: Dict, strict: bool = False) -> Tuple[Dict, List[str]]:
        """
        Validate all parameters in a parameter dictionary.
        
        Parameters:
        -----------
        params : Dict
            Parameter dictionary
        strict : bool
            Whether to enforce strict validation
            
        Returns:
        --------
        Tuple[Dict, List[str]]
            (validated_params, all_warnings)
        """
        validated_params = params.copy()
        all_warnings = []
        
        # Validate μ if present
        if 'mu' in params:
            validated_params['mu'], warnings = self.validate_mu(params['mu'], strict)
            all_warnings.extend(warnings)
        
        # Validate τ if present
        if 'tau' in params:
            validated_params['tau'], warnings = self.validate_tau(params['tau'], strict)
            all_warnings.extend(warnings)
        
        # Validate amplitude if present
        if 'amplitude' in params:
            amp_min, amp_max = self.safe_ranges['amplitude']
            if not np.isfinite(params['amplitude']) or params['amplitude'] <= 0:
                validated_params['amplitude'] = 1.0
                all_warnings.append(f"Invalid amplitude corrected to 1.0")
            elif params['amplitude'] < amp_min and not strict:
                validated_params['amplitude'] = amp_min
                all_warnings.append(f"Amplitude below safe range, corrected to {amp_min}")
            elif params['amplitude'] > amp_max and not strict:
                validated_params['amplitude'] = amp_max
                all_warnings.append(f"Amplitude above safe range, corrected to {amp_max}")
        
        return validated_params, all_warnings

class RobustSincCalculator:
    """
    Numerically robust sinc(πμ) calculation with Taylor expansion.
    
    Addresses UQ concerns:
    - Numerical instability for small μ values
    - Division by zero errors
    - Loss of precision in standard sinc calculation
    """
    
    def __init__(self, taylor_threshold: float = 1e-6):
        """
        Initialize robust sinc calculator.
        
        Parameters:
        -----------
        taylor_threshold : float
            Threshold below which to use Taylor expansion
        """
        self.taylor_threshold = taylor_threshold
    
    def sinc_robust(self, pi_mu: float) -> float:
        """
        Calculate sinc(πμ) with robust numerical handling.
        
        For |πμ| < threshold: sinc(πμ) ≈ 1 - (πμ)²/6 + (πμ)⁴/120 - ...
        For |πμ| ≥ threshold: sinc(πμ) = sin(πμ)/(πμ)
        
        Parameters:
        -----------
        pi_mu : float
            The argument πμ
            
        Returns:
        --------
        float
            Robust sinc(πμ) value
        """
        if not np.isfinite(pi_mu):
            return 1.0  # Safe fallback
        
        abs_pi_mu = abs(pi_mu)
        
        if abs_pi_mu < self.taylor_threshold:
            # Use Taylor expansion: sinc(x) = 1 - x²/6 + x⁴/120 - x⁶/5040 + ...
            x_squared = pi_mu * pi_mu
            sinc_value = 1.0 - x_squared/6.0 + x_squared*x_squared/120.0
            
            # Higher order terms for better precision
            if abs_pi_mu > self.taylor_threshold / 10:
                x_sixth = x_squared * x_squared * x_squared
                sinc_value -= x_sixth / 5040.0
                
            return sinc_value
        else:
            # Standard calculation for larger values
            return np.sin(pi_mu) / pi_mu if pi_mu != 0 else 1.0
    
    def sinc_enhancement_factor(self, mu: float) -> float:
        """
        Calculate sinc(πμ) enhancement factor with validation.
        
        Parameters:
        -----------
        mu : float
            Polymer parameter
            
        Returns:
        --------
        float
            Enhancement factor
        """
        # Validate input
        validator = RobustParameterValidator()
        validated_mu, warnings = validator.validate_mu(mu, strict=False)
        
        if warnings:
            logger.debug(f"Parameter validation warnings: {warnings}")
        
        return self.sinc_robust(np.pi * validated_mu)

class MultiStartOptimizer:
    """
    Multi-start optimization with convergence monitoring.
    
    Addresses UQ concerns:
    - Optimization convergence failures
    - Local minima trapping
    - Poor initial condition sensitivity
    """
    
    def __init__(self, n_starts: int = 10, convergence_tolerance: float = 1e-6):
        """
        Initialize multi-start optimizer.
        
        Parameters:
        -----------
        n_starts : int
            Number of random starting points
        convergence_tolerance : float
            Convergence tolerance
        """
        self.n_starts = n_starts
        self.convergence_tolerance = convergence_tolerance
        self.validator = RobustParameterValidator()
    
    def optimize_robust(self, objective_func: Callable, 
                       parameter_bounds: Dict[str, Tuple[float, float]],
                       method: str = 'L-BFGS-B') -> Dict:
        """
        Perform robust multi-start optimization.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to optimize
        parameter_bounds : Dict
            Parameter bounds for optimization
        method : str
            Optimization method
            
        Returns:
        --------
        Dict
            Optimization results with convergence analysis
        """
        results = []
        successful_optimizations = 0
        
        # Extract bounds for scipy
        param_names = list(parameter_bounds.keys())
        bounds = [parameter_bounds[name] for name in param_names]
        
        for start_idx in range(self.n_starts):
            try:
                # Generate random initial point within bounds
                x0 = []
                for name in param_names:
                    bound_min, bound_max = parameter_bounds[name]
                    x0.append(np.random.uniform(bound_min, bound_max))
                
                # Wrapper function for parameter validation
                def validated_objective(x):
                    params = dict(zip(param_names, x))
                    validated_params, warnings = self.validator.validate_all_parameters(params, strict=False)
                    validated_x = [validated_params[name] for name in param_names]
                    return objective_func(validated_x)
                
                # Perform optimization
                result = minimize(validated_objective, x0, method=method, bounds=bounds,
                                options={'ftol': self.convergence_tolerance})
                
                # Validate result
                if result.success and np.isfinite(result.fun):
                    successful_optimizations += 1
                    
                    # Convert back to parameter dictionary
                    optimal_params = dict(zip(param_names, result.x))
                    
                    results.append({
                        'start_idx': start_idx,
                        'success': True,
                        'optimal_params': optimal_params,
                        'optimal_value': result.fun,
                        'n_iterations': result.nit if hasattr(result, 'nit') else None,
                        'message': result.message
                    })
                else:
                    results.append({
                        'start_idx': start_idx,
                        'success': False,
                        'message': result.message,
                        'status': result.status if hasattr(result, 'status') else None
                    })
                    
            except Exception as e:
                results.append({
                    'start_idx': start_idx,
                    'success': False,
                    'error': str(e)
                })
        
        # Find best result
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['optimal_value'])
            convergence_rate = successful_optimizations / self.n_starts
        else:
            best_result = None
            convergence_rate = 0.0
        
        return {
            'best_result': best_result,
            'all_results': results,
            'convergence_rate': convergence_rate,
            'successful_optimizations': successful_optimizations,
            'total_starts': self.n_starts,
            'optimization_successful': best_result is not None
        }

class RobustNegativeEnergyGenerator:
    """
    Enhanced negative energy generator with robust optimization and validation.
    
    Replaces the original NegativeEnergyGenerator with improved UQ handling.
    """
    
    def __init__(self, mu: float = 0.7, tau: float = 1.0):
        """
        Initialize robust negative energy generator.
        
        Parameters:
        -----------
        mu : float
            Polymer parameter (will be validated)
        tau : float
            Timescale parameter (will be validated)
        """
        self.validator = RobustParameterValidator()
        self.sinc_calc = RobustSincCalculator()
        self.optimizer = MultiStartOptimizer()
        
        # Validate and store parameters
        self.mu, mu_warnings = self.validator.validate_mu(mu, strict=False)
        self.tau, tau_warnings = self.validator.validate_tau(tau, strict=False)
        
        all_warnings = mu_warnings + tau_warnings
        if all_warnings:
            logger.warning(f"Parameter validation warnings: {all_warnings}")
        
        # Initialize quantum inequality bounds with validated parameters
        from ..optimization.quantum_inequality import QuantumInequalityBounds
        self.qi_bounds = QuantumInequalityBounds(mu=self.mu, tau=self.tau)
    
    def energy_density_profile_robust(self, t: np.ndarray, 
                                    amplitude: float = 1.0,
                                    profile_type: str = 'optimal') -> np.ndarray:
        """
        Generate negative energy density profile with robust calculation.
        
        Parameters:
        -----------
        t : np.ndarray
            Time array
        amplitude : float
            Energy density amplitude (will be validated)
        profile_type : str
            Profile type ('optimal', 'gaussian')
            
        Returns:
        --------
        np.ndarray
            Robust energy density profile
        """
        # Validate amplitude
        validated_amplitude, warnings = self.validator.validate_all_parameters(
            {'amplitude': amplitude}, strict=False)
        amplitude = validated_amplitude['amplitude']
        
        if warnings:
            logger.debug(f"Amplitude validation warnings: {warnings}")
        
        # Calculate base profile
        if profile_type == 'optimal':
            base_profile = -amplitude * self.qi_bounds.optimal_sampling_function(t)
        elif profile_type == 'gaussian':
            base_profile = -amplitude * self.qi_bounds.gaussian_sampling_function(t)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        # Apply robust sinc enhancement
        sinc_enhancement = self.sinc_calc.sinc_enhancement_factor(self.mu)
        
        return base_profile * sinc_enhancement
    
    def optimize_robust_extraction(self, t_range: Tuple[float, float],
                                 n_points: int = 1000) -> Dict:
        """
        Optimize negative energy extraction with robust methods.
        
        Parameters:
        -----------
        t_range : Tuple[float, float]
            Time range for optimization
        n_points : int
            Number of time points
            
        Returns:
        --------
        Dict
            Robust optimization results
        """
        # Validate inputs
        if not all(np.isfinite(t_range)) or t_range[1] <= t_range[0]:
            raise ValueError(f"Invalid time range: {t_range}")
        
        n_points = max(10, min(10000, int(n_points)))  # Safe bounds
        
        t = np.linspace(t_range[0], t_range[1], n_points)
        
        # Define optimization objective
        def amplitude_objective(params):
            """Objective: maximize extractable energy within bounds."""
            amplitude = params[0]
            
            try:
                # Generate energy profile
                rho_eff = self.energy_density_profile_robust(t, amplitude=amplitude)
                f_sampling = self.qi_bounds.optimal_sampling_function(t)
                
                # Calculate extraction integral
                integrand = rho_eff * f_sampling
                integral_value = np.trapz(integrand, t)
                
                # Get quantum bound
                enhanced_bound = self.qi_bounds.enhanced_ford_roman_bound()
                
                # Check if within bounds (with safety margin)
                safety_factor = 0.95
                if integral_value >= enhanced_bound * safety_factor:
                    # Valid: return negative energy (to maximize)
                    return -abs(integral_value)
                else:
                    # Invalid: return penalty
                    violation = abs(integral_value) / abs(enhanced_bound)
                    penalty = 1e6 * (violation - 0.95)  # Large penalty for violations
                    return penalty
                    
            except Exception as e:
                logger.debug(f"Objective evaluation failed: {e}")
                return 1e6  # Large penalty for failures
        
        # Set up optimization bounds
        amp_min, amp_max = self.validator.safe_ranges['amplitude']
        parameter_bounds = {'amplitude': (amp_min, amp_max)}
        
        # Perform robust optimization
        opt_result = self.optimizer.optimize_robust(
            amplitude_objective, parameter_bounds, method='L-BFGS-B')
        
        if opt_result['optimization_successful']:
            best_amplitude = opt_result['best_result']['optimal_params']['amplitude']
            
            # Generate final optimized profile
            rho_optimized = self.energy_density_profile_robust(t, amplitude=best_amplitude)
            f_optimal = self.qi_bounds.optimal_sampling_function(t)
            
            # Final validation
            from ..optimization.quantum_inequality import NegativeEnergyGenerator
            legacy_generator = NegativeEnergyGenerator(mu=self.mu, tau=self.tau)
            final_validation = legacy_generator.validate_quantum_inequality(
                t, rho_optimized, f_optimal)
            
            extracted_energy = abs(final_validation['integral_value'])
        else:
            best_amplitude = 0.0
            rho_optimized = np.zeros_like(t)
            f_optimal = self.qi_bounds.optimal_sampling_function(t)
            final_validation = {'is_valid': False, 'integral_value': 0.0}
            extracted_energy = 0.0
        
        return {
            'time_array': t,
            'optimal_energy_density': rho_optimized,
            'sampling_function': f_optimal,
            'max_amplitude': best_amplitude,
            'extracted_energy': extracted_energy,
            'validation_results': final_validation,
            'optimization_results': opt_result,
            'optimization_success': opt_result['optimization_successful'],
            'convergence_rate': opt_result['convergence_rate']
        }

# Factory function for backward compatibility
def create_robust_negative_energy_generator(mu: float = 0.7, tau: float = 1.0) -> RobustNegativeEnergyGenerator:
    """
    Factory function to create robust negative energy generator.
    
    This replaces the original NegativeEnergyGenerator for improved UQ handling.
    """
    return RobustNegativeEnergyGenerator(mu=mu, tau=tau)


if __name__ == "__main__":
    print("🔧 Robust Optimization Framework - Testing")
    print("=" * 50)
    
    # Test parameter validation
    print("\n📋 Testing Parameter Validation...")
    validator = RobustParameterValidator()
    
    # Test various parameter values
    test_params = [
        {'mu': 0.7, 'tau': 1.0},     # Normal
        {'mu': -0.1, 'tau': 1e-5},   # Invalid μ, τ too small
        {'mu': 5.0, 'tau': 1000.0},  # μ too large, τ very large
        {'mu': np.nan, 'tau': np.inf} # Non-finite values
    ]
    
    for i, params in enumerate(test_params):
        print(f"  Test {i+1}: {params}")
        validated, warnings = validator.validate_all_parameters(params, strict=False)
        print(f"    Validated: {validated}")
        if warnings:
            print(f"    Warnings: {warnings}")
    
    # Test robust sinc calculation
    print("\n🔄 Testing Robust Sinc Calculation...")
    sinc_calc = RobustSincCalculator()
    
    test_mu_values = [0.0, 1e-8, 1e-4, 0.7, 1.0, 2.0]
    for mu in test_mu_values:
        sinc_value = sinc_calc.sinc_enhancement_factor(mu)
        print(f"  μ={mu:8.1e}: sinc(πμ)={sinc_value:.8f}")
    
    # Test robust optimization
    print("\n🎯 Testing Robust Optimization...")
    robust_generator = RobustNegativeEnergyGenerator(mu=0.7, tau=1.0)
    
    optimization_result = robust_generator.optimize_robust_extraction((-5, 5), n_points=100)
    
    print(f"  Optimization Success: {optimization_result['optimization_success']}")
    print(f"  Convergence Rate: {optimization_result['convergence_rate']:.1%}")
    print(f"  Max Amplitude: {optimization_result['max_amplitude']:.6f}")
    print(f"  Extracted Energy: {optimization_result['extracted_energy']:.6e}")
    
    print("\n✅ Robust Framework Testing Complete!")
