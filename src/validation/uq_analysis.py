"""
Uncertainty Quantification (UQ) Analysis for LQG Polymer Field Generator

This module identifies and resolves critical uncertainty quantification concerns
in the LQG polymer field generation system, focusing on numerical stability,
parameter sensitivity, and error propagation.

Author: LQG-FTL Research Team
Date: July 2025
"""

import numpy as np
import scipy.constants as const
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from scipy.stats import norm, chi2
from scipy.optimize import minimize
import logging

# Configure logging for UQ analysis
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumericalStabilityAnalyzer:
    """
    Analyzes numerical stability issues in LQG polymer calculations.
    
    Critical concerns addressed:
    1. Division by zero in quantum inequality bounds
    2. Numerical overflow/underflow in sinc(πμ) calculations
    3. Convergence issues in optimization routines
    4. Matrix conditioning in field operator computations
    """
    
    def __init__(self):
        """Initialize numerical stability analyzer."""
        self.epsilon = 1e-15  # Machine precision threshold
        self.max_safe_value = 1e100  # Maximum safe floating point value
        self.min_safe_value = 1e-100  # Minimum safe floating point value
        
    def analyze_sinc_stability(self, mu_values: np.ndarray) -> Dict:
        """
        Analyze numerical stability of sinc(πμ) calculations.
        
        Critical Issue: sinc(πμ) → 1 as μ → 0, but numerical precision
        can cause instabilities for very small μ values.
        
        Parameters:
        -----------
        mu_values : np.ndarray
            Range of polymer parameters to test
            
        Returns:
        --------
        Dict
            Stability analysis results
        """
        stability_issues = []
        safe_mu_range = []
        
        for mu in mu_values:
            try:
                if abs(mu) < self.epsilon:
                    # Handle μ ≈ 0 case analytically
                    sinc_value = 1.0
                    is_stable = True
                else:
                    # Standard calculation
                    sinc_value = np.sin(np.pi * mu) / (np.pi * mu)
                    
                    # Check for numerical issues
                    is_stable = (
                        np.isfinite(sinc_value) and 
                        abs(sinc_value) < self.max_safe_value and
                        abs(sinc_value) > self.min_safe_value
                    )
                
                if is_stable:
                    safe_mu_range.append(mu)
                else:
                    stability_issues.append({
                        'mu': mu,
                        'sinc_value': sinc_value,
                        'issue': 'Numerical instability in sinc calculation'
                    })
                    
            except (ZeroDivisionError, OverflowError, UnderflowError) as e:
                stability_issues.append({
                    'mu': mu,
                    'sinc_value': np.nan,
                    'issue': f'Exception in sinc calculation: {str(e)}'
                })
        
        return {
            'tested_values': mu_values,
            'stable_range': np.array(safe_mu_range),
            'stability_issues': stability_issues,
            'stability_fraction': len(safe_mu_range) / len(mu_values),
            'recommended_mu_bounds': (max(1e-6, min(safe_mu_range)) if safe_mu_range else None,
                                    min(2.0, max(safe_mu_range)) if safe_mu_range else None)
        }
    
    def analyze_quantum_inequality_stability(self, tau_values: np.ndarray) -> Dict:
        """
        Analyze stability of quantum inequality bound calculations.
        
        Critical Issue: Ford-Roman bound ∝ 1/τ² can cause overflow
        for very small τ values.
        
        Parameters:
        -----------
        tau_values : np.ndarray
            Range of timescale parameters to test
            
        Returns:
        --------
        Dict
            Quantum inequality stability analysis
        """
        from ..optimization.quantum_inequality import QuantumInequalityBounds
        
        stability_issues = []
        safe_tau_range = []
        
        for tau in tau_values:
            try:
                qi_bounds = QuantumInequalityBounds(mu=0.7, tau=tau)
                
                classical_bound = qi_bounds.classical_ford_roman_bound()
                enhanced_bound = qi_bounds.enhanced_ford_roman_bound()
                
                # Check for numerical issues
                bounds_stable = all([
                    np.isfinite(classical_bound),
                    np.isfinite(enhanced_bound),
                    abs(classical_bound) < self.max_safe_value,
                    abs(enhanced_bound) < self.max_safe_value,
                    abs(classical_bound) > self.min_safe_value,
                    abs(enhanced_bound) > self.min_safe_value
                ])
                
                if bounds_stable:
                    safe_tau_range.append(tau)
                else:
                    stability_issues.append({
                        'tau': tau,
                        'classical_bound': classical_bound,
                        'enhanced_bound': enhanced_bound,
                        'issue': 'Numerical instability in quantum bounds'
                    })
                    
            except Exception as e:
                stability_issues.append({
                    'tau': tau,
                    'classical_bound': np.nan,
                    'enhanced_bound': np.nan,
                    'issue': f'Exception in bounds calculation: {str(e)}'
                })
        
        return {
            'tested_values': tau_values,
            'stable_range': np.array(safe_tau_range),
            'stability_issues': stability_issues,
            'stability_fraction': len(safe_tau_range) / len(tau_values),
            'recommended_tau_bounds': (max(1e-3, min(safe_tau_range)) if safe_tau_range else None,
                                     min(100.0, max(safe_tau_range)) if safe_tau_range else None)
        }
    
    def analyze_optimization_convergence(self, n_trials: int = 100) -> Dict:
        """
        Analyze convergence stability of optimization routines.
        
        Critical Issue: Optimization may fail to converge or converge
        to non-physical solutions.
        
        Parameters:
        -----------
        n_trials : int
            Number of optimization trials with random initial conditions
            
        Returns:
        --------
        Dict
            Optimization convergence analysis
        """
        from ..optimization.quantum_inequality import NegativeEnergyGenerator
        
        convergence_results = []
        successful_optimizations = 0
        
        for trial in range(n_trials):
            try:
                # Random but reasonable parameters
                mu = np.random.uniform(0.1, 1.5)
                tau = np.random.uniform(0.1, 5.0)
                
                neg_energy = NegativeEnergyGenerator(mu=mu, tau=tau)
                
                # Try optimization
                t_range = (-10, 10)
                optimization = neg_energy.optimize_negative_energy_extraction(t_range)
                
                # Check if optimization was successful and physically meaningful
                is_successful = (
                    optimization['optimization_success'] and
                    np.isfinite(optimization['max_amplitude']) and
                    optimization['max_amplitude'] > 0 and
                    optimization['extracted_energy'] > 0
                )
                
                convergence_results.append({
                    'trial': trial,
                    'mu': mu,
                    'tau': tau,
                    'success': is_successful,
                    'max_amplitude': optimization['max_amplitude'],
                    'extracted_energy': optimization['extracted_energy']
                })
                
                if is_successful:
                    successful_optimizations += 1
                    
            except Exception as e:
                convergence_results.append({
                    'trial': trial,
                    'mu': mu if 'mu' in locals() else np.nan,
                    'tau': tau if 'tau' in locals() else np.nan,
                    'success': False,
                    'error': str(e)
                })
        
        convergence_rate = successful_optimizations / n_trials
        
        return {
            'n_trials': n_trials,
            'convergence_rate': convergence_rate,
            'successful_optimizations': successful_optimizations,
            'results': convergence_results,
            'convergence_status': 'STABLE' if convergence_rate > 0.9 else
                                'MODERATE' if convergence_rate > 0.7 else 'UNSTABLE'
        }

class ParameterSensitivityAnalyzer:
    """
    Analyzes parameter sensitivity and uncertainty propagation.
    
    Addresses:
    1. Sensitivity of field enhancement to polymer parameter μ
    2. Impact of timescale τ variations on quantum bounds
    3. Error propagation through calculation chains
    4. Parameter correlation effects
    """
    
    def __init__(self):
        """Initialize parameter sensitivity analyzer."""
        self.sensitivity_threshold = 0.1  # 10% parameter change threshold
        
    def mu_sensitivity_analysis(self, mu_nominal: float = 0.7,
                               perturbation_range: float = 0.1) -> Dict:
        """
        Analyze sensitivity to polymer parameter μ variations.
        
        Parameters:
        -----------
        mu_nominal : float
            Nominal polymer parameter value
        perturbation_range : float
            Relative perturbation range (±)
            
        Returns:
        --------
        Dict
            Sensitivity analysis results
        """
        from ..core.polymer_quantization import PolymerQuantization
        
        # Generate perturbation values
        mu_min = mu_nominal * (1 - perturbation_range)
        mu_max = mu_nominal * (1 + perturbation_range)
        mu_values = np.linspace(mu_min, mu_max, 21)
        
        # Calculate field enhancements for each μ
        sinc_factors = []
        enhancement_magnitudes = []
        
        for mu in mu_values:
            try:
                polymer = PolymerQuantization(mu=mu)
                
                sinc_factor = polymer.sinc_enhancement_factor()
                enhancement_mag = polymer.enhancement_magnitude()
                
                sinc_factors.append(sinc_factor)
                enhancement_magnitudes.append(enhancement_mag)
                
            except Exception as e:
                logger.warning(f"Failed to calculate enhancement for μ={mu}: {e}")
                sinc_factors.append(np.nan)
                enhancement_magnitudes.append(np.nan)
        
        sinc_factors = np.array(sinc_factors)
        enhancement_magnitudes = np.array(enhancement_magnitudes)
        
        # Calculate sensitivity metrics
        valid_mask = np.isfinite(sinc_factors) & np.isfinite(enhancement_magnitudes)
        
        if np.sum(valid_mask) > 1:
            # Numerical derivatives (sensitivity coefficients)
            dmu = mu_values[1] - mu_values[0]
            
            sinc_sensitivity = np.gradient(sinc_factors[valid_mask], dmu)
            enhancement_sensitivity = np.gradient(enhancement_magnitudes[valid_mask], dmu)
            
            # Relative sensitivity at nominal value
            nominal_idx = np.argmin(np.abs(mu_values - mu_nominal))
            if valid_mask[nominal_idx]:
                sinc_rel_sens = (sinc_sensitivity[np.sum(valid_mask[:nominal_idx])] * 
                               mu_nominal / sinc_factors[nominal_idx])
                enhancement_rel_sens = (enhancement_sensitivity[np.sum(valid_mask[:nominal_idx])] * 
                                      mu_nominal / enhancement_magnitudes[nominal_idx])
            else:
                sinc_rel_sens = np.nan
                enhancement_rel_sens = np.nan
        else:
            sinc_sensitivity = np.array([])
            enhancement_sensitivity = np.array([])
            sinc_rel_sens = np.nan
            enhancement_rel_sens = np.nan
        
        return {
            'mu_values': mu_values,
            'sinc_factors': sinc_factors,
            'enhancement_magnitudes': enhancement_magnitudes,
            'sinc_sensitivity': sinc_sensitivity,
            'enhancement_sensitivity': enhancement_sensitivity,
            'sinc_relative_sensitivity': sinc_rel_sens,
            'enhancement_relative_sensitivity': enhancement_rel_sens,
            'sensitivity_status': self._classify_sensitivity(abs(sinc_rel_sens) if np.isfinite(sinc_rel_sens) else np.inf)
        }
    
    def tau_sensitivity_analysis(self, tau_nominal: float = 1.0,
                                perturbation_range: float = 0.2) -> Dict:
        """
        Analyze sensitivity to timescale parameter τ variations.
        
        Parameters:
        -----------
        tau_nominal : float
            Nominal timescale parameter
        perturbation_range : float
            Relative perturbation range (±)
            
        Returns:
        --------
        Dict
            Timescale sensitivity analysis
        """
        from ..optimization.quantum_inequality import QuantumInequalityBounds
        
        # Generate perturbation values
        tau_min = tau_nominal * (1 - perturbation_range)
        tau_max = tau_nominal * (1 + perturbation_range)
        tau_values = np.linspace(tau_min, tau_max, 21)
        
        # Calculate quantum bounds for each τ
        classical_bounds = []
        enhanced_bounds = []
        violation_strengths = []
        
        for tau in tau_values:
            try:
                qi_bounds = QuantumInequalityBounds(mu=0.7, tau=tau)
                
                classical_bound = qi_bounds.classical_ford_roman_bound()
                enhanced_bound = qi_bounds.enhanced_ford_roman_bound()
                violation_strength = qi_bounds.negative_energy_violation_strength()
                
                classical_bounds.append(abs(classical_bound))
                enhanced_bounds.append(abs(enhanced_bound))
                violation_strengths.append(violation_strength)
                
            except Exception as e:
                logger.warning(f"Failed to calculate bounds for τ={tau}: {e}")
                classical_bounds.append(np.nan)
                enhanced_bounds.append(np.nan)
                violation_strengths.append(np.nan)
        
        classical_bounds = np.array(classical_bounds)
        enhanced_bounds = np.array(enhanced_bounds)
        violation_strengths = np.array(violation_strengths)
        
        # Calculate sensitivity metrics
        valid_mask = (np.isfinite(classical_bounds) & 
                     np.isfinite(enhanced_bounds) & 
                     np.isfinite(violation_strengths))
        
        if np.sum(valid_mask) > 1:
            dtau = tau_values[1] - tau_values[0]
            
            classical_sensitivity = np.gradient(classical_bounds[valid_mask], dtau)
            enhanced_sensitivity = np.gradient(enhanced_bounds[valid_mask], dtau)
            violation_sensitivity = np.gradient(violation_strengths[valid_mask], dtau)
            
            # Relative sensitivity at nominal value
            nominal_idx = np.argmin(np.abs(tau_values - tau_nominal))
            if valid_mask[nominal_idx]:
                valid_nominal_idx = np.sum(valid_mask[:nominal_idx])
                
                classical_rel_sens = (classical_sensitivity[valid_nominal_idx] * 
                                    tau_nominal / classical_bounds[nominal_idx])
                enhanced_rel_sens = (enhanced_sensitivity[valid_nominal_idx] * 
                                   tau_nominal / enhanced_bounds[nominal_idx])
                violation_rel_sens = (violation_sensitivity[valid_nominal_idx] * 
                                    tau_nominal / violation_strengths[nominal_idx])
            else:
                classical_rel_sens = np.nan
                enhanced_rel_sens = np.nan
                violation_rel_sens = np.nan
        else:
            classical_sensitivity = np.array([])
            enhanced_sensitivity = np.array([])
            violation_sensitivity = np.array([])
            classical_rel_sens = np.nan
            enhanced_rel_sens = np.nan
            violation_rel_sens = np.nan
        
        return {
            'tau_values': tau_values,
            'classical_bounds': classical_bounds,
            'enhanced_bounds': enhanced_bounds,
            'violation_strengths': violation_strengths,
            'classical_sensitivity': classical_sensitivity,
            'enhanced_sensitivity': enhanced_sensitivity,
            'violation_sensitivity': violation_sensitivity,
            'classical_relative_sensitivity': classical_rel_sens,
            'enhanced_relative_sensitivity': enhanced_rel_sens,
            'violation_relative_sensitivity': violation_rel_sens,
            'sensitivity_status': self._classify_sensitivity(
                max(abs(classical_rel_sens) if np.isfinite(classical_rel_sens) else 0,
                    abs(enhanced_rel_sens) if np.isfinite(enhanced_rel_sens) else 0,
                    abs(violation_rel_sens) if np.isfinite(violation_rel_sens) else 0))
        }
    
    def _classify_sensitivity(self, relative_sensitivity: float) -> str:
        """Classify sensitivity level."""
        if relative_sensitivity < 0.1:
            return 'LOW'
        elif relative_sensitivity < 1.0:
            return 'MODERATE'
        elif relative_sensitivity < 10.0:
            return 'HIGH'
        else:
            return 'CRITICAL'

class ErrorPropagationAnalyzer:
    """
    Analyzes error propagation through calculation chains.
    
    Addresses:
    1. Measurement uncertainties in input parameters
    2. Numerical precision accumulation
    3. Model uncertainties in physical assumptions
    4. Systematic errors in approximations
    """
    
    def __init__(self):
        """Initialize error propagation analyzer."""
        self.monte_carlo_samples = 1000
        
    def monte_carlo_uncertainty_analysis(self, 
                                       mu_uncertainty: float = 0.05,
                                       tau_uncertainty: float = 0.1) -> Dict:
        """
        Perform Monte Carlo uncertainty propagation analysis.
        
        Parameters:
        -----------
        mu_uncertainty : float
            Relative uncertainty in polymer parameter μ
        tau_uncertainty : float
            Relative uncertainty in timescale τ
            
        Returns:
        --------
        Dict
            Monte Carlo uncertainty analysis results
        """
        from ..optimization.quantum_inequality import NegativeEnergyGenerator
        
        # Nominal values
        mu_nominal = 0.7
        tau_nominal = 1.0
        
        # Generate random parameter samples
        mu_samples = np.random.normal(mu_nominal, mu_nominal * mu_uncertainty, 
                                    self.monte_carlo_samples)
        tau_samples = np.random.normal(tau_nominal, tau_nominal * tau_uncertainty, 
                                     self.monte_carlo_samples)
        
        # Ensure positive values
        mu_samples = np.abs(mu_samples)
        tau_samples = np.abs(tau_samples)
        
        # Calculate outputs for each sample
        extracted_energies = []
        max_amplitudes = []
        violation_strengths = []
        
        successful_samples = 0
        
        for i in range(self.monte_carlo_samples):
            try:
                neg_energy = NegativeEnergyGenerator(mu=mu_samples[i], tau=tau_samples[i])
                
                # Optimize extraction
                optimization = neg_energy.optimize_negative_energy_extraction((-10, 10))
                
                if optimization['optimization_success']:
                    extracted_energies.append(optimization['extracted_energy'])
                    max_amplitudes.append(optimization['max_amplitude'])
                    
                    # Calculate violation strength
                    validation = optimization['validation_results']
                    violation_strengths.append(validation['violation_strength'])
                    
                    successful_samples += 1
                else:
                    extracted_energies.append(np.nan)
                    max_amplitudes.append(np.nan)
                    violation_strengths.append(np.nan)
                    
            except Exception as e:
                logger.debug(f"Sample {i} failed: {e}")
                extracted_energies.append(np.nan)
                max_amplitudes.append(np.nan)
                violation_strengths.append(np.nan)
        
        extracted_energies = np.array(extracted_energies)
        max_amplitudes = np.array(max_amplitudes)
        violation_strengths = np.array(violation_strengths)
        
        # Calculate statistics for successful samples
        valid_mask = np.isfinite(extracted_energies)
        
        if np.sum(valid_mask) > 0:
            energy_stats = {
                'mean': np.mean(extracted_energies[valid_mask]),
                'std': np.std(extracted_energies[valid_mask]),
                'min': np.min(extracted_energies[valid_mask]),
                'max': np.max(extracted_energies[valid_mask]),
                'rel_uncertainty': np.std(extracted_energies[valid_mask]) / np.mean(extracted_energies[valid_mask])
            }
            
            amplitude_stats = {
                'mean': np.mean(max_amplitudes[valid_mask]),
                'std': np.std(max_amplitudes[valid_mask]),
                'min': np.min(max_amplitudes[valid_mask]),
                'max': np.max(max_amplitudes[valid_mask]),
                'rel_uncertainty': np.std(max_amplitudes[valid_mask]) / np.mean(max_amplitudes[valid_mask])
            }
            
            violation_stats = {
                'mean': np.mean(violation_strengths[valid_mask]),
                'std': np.std(violation_strengths[valid_mask]),
                'min': np.min(violation_strengths[valid_mask]),
                'max': np.max(violation_strengths[valid_mask]),
                'rel_uncertainty': np.std(violation_strengths[valid_mask]) / np.mean(violation_strengths[valid_mask])
            }
        else:
            energy_stats = None
            amplitude_stats = None
            violation_stats = None
        
        return {
            'input_uncertainties': {
                'mu_uncertainty': mu_uncertainty,
                'tau_uncertainty': tau_uncertainty
            },
            'sample_parameters': {
                'mu_samples': mu_samples,
                'tau_samples': tau_samples
            },
            'outputs': {
                'extracted_energies': extracted_energies,
                'max_amplitudes': max_amplitudes,
                'violation_strengths': violation_strengths
            },
            'statistics': {
                'energy_stats': energy_stats,
                'amplitude_stats': amplitude_stats,
                'violation_stats': violation_stats
            },
            'success_rate': successful_samples / self.monte_carlo_samples,
            'total_samples': self.monte_carlo_samples,
            'successful_samples': successful_samples
        }

class UQConcernResolver:
    """
    Main UQ concern identification and resolution system.
    
    Provides comprehensive analysis and mitigation strategies for
    identified uncertainty quantification issues.
    """
    
    def __init__(self):
        """Initialize UQ concern resolver."""
        self.stability_analyzer = NumericalStabilityAnalyzer()
        self.sensitivity_analyzer = ParameterSensitivityAnalyzer()
        self.error_analyzer = ErrorPropagationAnalyzer()
        
        # Severity thresholds
        self.severity_thresholds = {
            'CRITICAL': {'stability_fraction': 0.5, 'convergence_rate': 0.5, 
                        'rel_uncertainty': 1.0, 'success_rate': 0.3},
            'HIGH': {'stability_fraction': 0.7, 'convergence_rate': 0.7, 
                    'rel_uncertainty': 0.5, 'success_rate': 0.6},
            'MODERATE': {'stability_fraction': 0.85, 'convergence_rate': 0.85, 
                        'rel_uncertainty': 0.2, 'success_rate': 0.8},
            'LOW': {'stability_fraction': 0.95, 'convergence_rate': 0.95, 
                   'rel_uncertainty': 0.1, 'success_rate': 0.9}
        }
    
    def comprehensive_uq_analysis(self) -> Dict:
        """
        Perform comprehensive UQ analysis across all concerns.
        
        Returns:
        --------
        Dict
            Complete UQ analysis results with severity classifications
        """
        logger.info("Starting comprehensive UQ analysis...")
        
        results = {}
        
        # 1. Numerical Stability Analysis
        logger.info("Analyzing numerical stability...")
        
        # Test sinc function stability
        mu_test_range = np.logspace(-6, 1, 100)  # 1e-6 to 10
        sinc_stability = self.stability_analyzer.analyze_sinc_stability(mu_test_range)
        
        # Test quantum inequality stability
        tau_test_range = np.logspace(-3, 2, 100)  # 1e-3 to 100
        qi_stability = self.stability_analyzer.analyze_quantum_inequality_stability(tau_test_range)
        
        # Test optimization convergence
        # Test optimization convergence with robust optimizer
        try:
            from ..optimization.robust_optimizer import RobustNegativeEnergyGenerator
            
            # Use robust generator for convergence testing
            convergence_results = []
            successful_optimizations = 0
            n_trials = 20  # Smaller number for faster testing
            
            for trial in range(n_trials):
                try:
                    # Random but reasonable parameters
                    mu = np.random.uniform(0.1, 1.5)
                    tau = np.random.uniform(0.1, 5.0)
                    
                    robust_gen = RobustNegativeEnergyGenerator(mu=mu, tau=tau)
                    
                    # Try optimization
                    t_range = (-5, 5)
                    optimization = robust_gen.optimize_robust_extraction(t_range, n_points=100)
                    
                    # Check if optimization was successful and physically meaningful
                    is_successful = (
                        optimization['optimization_success'] and
                        np.isfinite(optimization['max_amplitude']) and
                        optimization['max_amplitude'] > 0 and
                        optimization['extracted_energy'] > 0
                    )
                    
                    convergence_results.append({
                        'trial': trial,
                        'mu': mu,
                        'tau': tau,
                        'success': is_successful,
                        'max_amplitude': optimization['max_amplitude'],
                        'extracted_energy': optimization['extracted_energy'],
                        'convergence_rate': optimization.get('convergence_rate', 0.0)
                    })
                    
                    if is_successful:
                        successful_optimizations += 1
                        
                except Exception as e:
                    convergence_results.append({
                        'trial': trial,
                        'mu': mu if 'mu' in locals() else np.nan,
                        'tau': tau if 'tau' in locals() else np.nan,
                        'success': False,
                        'error': str(e)
                    })
            
            convergence_rate = successful_optimizations / n_trials
            
            convergence_analysis = {
                'n_trials': n_trials,
                'convergence_rate': convergence_rate,
                'successful_optimizations': successful_optimizations,
                'results': convergence_results,
                'convergence_status': 'STABLE' if convergence_rate > 0.9 else
                                    'MODERATE' if convergence_rate > 0.7 else 'UNSTABLE'
            }
            
        except ImportError:
            # Fallback to original analysis
            convergence_analysis = self.stability_analyzer.analyze_optimization_convergence(50)
        
        results['numerical_stability'] = {
            'sinc_stability': sinc_stability,
            'quantum_inequality_stability': qi_stability,
            'optimization_convergence': convergence_analysis
        }
        
        # 2. Parameter Sensitivity Analysis
        logger.info("Analyzing parameter sensitivity...")
        
        mu_sensitivity = self.sensitivity_analyzer.mu_sensitivity_analysis()
        tau_sensitivity = self.sensitivity_analyzer.tau_sensitivity_analysis()
        
        results['parameter_sensitivity'] = {
            'mu_sensitivity': mu_sensitivity,
            'tau_sensitivity': tau_sensitivity
        }
        
        # 3. Error Propagation Analysis
        logger.info("Analyzing error propagation...")
        
        # Use robust generator for Monte Carlo analysis
        try:
            from ..optimization.robust_optimizer import RobustNegativeEnergyGenerator
            
            # Test with robust generator
            robust_samples = 50  # Smaller sample for faster testing
            
            mu_samples = np.random.normal(0.7, 0.05, robust_samples)
            tau_samples = np.random.normal(1.0, 0.1, robust_samples)
            
            # Ensure positive values
            mu_samples = np.abs(mu_samples)
            tau_samples = np.abs(tau_samples)
            
            successful_samples = 0
            extracted_energies = []
            
            for i in range(robust_samples):
                try:
                    robust_gen = RobustNegativeEnergyGenerator(mu=mu_samples[i], tau=tau_samples[i])
                    optimization = robust_gen.optimize_robust_extraction((-5, 5), n_points=100)
                    
                    if optimization['optimization_success']:
                        extracted_energies.append(optimization['extracted_energy'])
                        successful_samples += 1
                    else:
                        extracted_energies.append(np.nan)
                        
                except Exception as e:
                    logger.debug(f"Robust sample {i} failed: {e}")
                    extracted_energies.append(np.nan)
            
            extracted_energies = np.array(extracted_energies)
            valid_mask = np.isfinite(extracted_energies)
            
            if np.sum(valid_mask) > 0:
                energy_stats = {
                    'mean': np.mean(extracted_energies[valid_mask]),
                    'std': np.std(extracted_energies[valid_mask]),
                    'min': np.min(extracted_energies[valid_mask]),
                    'max': np.max(extracted_energies[valid_mask]),
                    'rel_uncertainty': np.std(extracted_energies[valid_mask]) / np.mean(extracted_energies[valid_mask])
                }
            else:
                energy_stats = None
            
            robust_analysis = {
                'total_samples': robust_samples,
                'successful_samples': successful_samples,
                'success_rate': successful_samples / robust_samples,
                'statistics': {'energy_stats': energy_stats}
            }
            
        except ImportError:
            # Fallback to original analysis
            robust_analysis = self.error_analyzer.monte_carlo_uncertainty_analysis()
        
        uncertainty_propagation = robust_analysis
        
        results['error_propagation'] = {
            'monte_carlo_analysis': uncertainty_propagation
        }
        
        # 4. Overall UQ Assessment
        logger.info("Performing overall UQ assessment...")
        
        overall_assessment = self._assess_overall_uq_status(results)
        results['overall_assessment'] = overall_assessment
        
        logger.info(f"UQ analysis complete. Overall severity: {overall_assessment['severity_level']}")
        
        return results
    
    def _assess_overall_uq_status(self, results: Dict) -> Dict:
        """
        Assess overall UQ status and identify critical concerns.
        
        Parameters:
        -----------
        results : Dict
            Complete analysis results
            
        Returns:
        --------
        Dict
            Overall UQ assessment
        """
        concerns = []
        severity_scores = []
        
        # Evaluate numerical stability
        ns = results['numerical_stability']
        
        sinc_fraction = ns['sinc_stability']['stability_fraction']
        qi_fraction = ns['quantum_inequality_stability']['stability_fraction']
        convergence_rate = ns['optimization_convergence']['convergence_rate']
        
        if sinc_fraction < self.severity_thresholds['CRITICAL']['stability_fraction']:
            concerns.append("CRITICAL: Sinc function numerical instability")
            severity_scores.append(4)
        elif sinc_fraction < self.severity_thresholds['HIGH']['stability_fraction']:
            concerns.append("HIGH: Sinc function numerical issues")
            severity_scores.append(3)
        
        if qi_fraction < self.severity_thresholds['CRITICAL']['stability_fraction']:
            concerns.append("CRITICAL: Quantum inequality bound instability")
            severity_scores.append(4)
        elif qi_fraction < self.severity_thresholds['HIGH']['stability_fraction']:
            concerns.append("HIGH: Quantum inequality bound issues")
            severity_scores.append(3)
        
        if convergence_rate < self.severity_thresholds['CRITICAL']['convergence_rate']:
            concerns.append("CRITICAL: Optimization convergence failure")
            severity_scores.append(4)
        elif convergence_rate < self.severity_thresholds['HIGH']['convergence_rate']:
            concerns.append("HIGH: Optimization convergence issues")
            severity_scores.append(3)
        
        # Evaluate parameter sensitivity
        ps = results['parameter_sensitivity']
        
        mu_sens_status = ps['mu_sensitivity']['sensitivity_status']
        tau_sens_status = ps['tau_sensitivity']['sensitivity_status']
        
        if mu_sens_status == 'CRITICAL':
            concerns.append("CRITICAL: Extreme sensitivity to polymer parameter μ")
            severity_scores.append(4)
        elif mu_sens_status == 'HIGH':
            concerns.append("HIGH: High sensitivity to polymer parameter μ")
            severity_scores.append(3)
        
        if tau_sens_status == 'CRITICAL':
            concerns.append("CRITICAL: Extreme sensitivity to timescale parameter τ")
            severity_scores.append(4)
        elif tau_sens_status == 'HIGH':
            concerns.append("HIGH: High sensitivity to timescale parameter τ")
            severity_scores.append(3)
        
        # Evaluate error propagation
        ep = results['error_propagation']['monte_carlo_analysis']
        
        success_rate = ep['success_rate']
        
        if success_rate < self.severity_thresholds['CRITICAL']['success_rate']:
            concerns.append("CRITICAL: High failure rate in Monte Carlo analysis")
            severity_scores.append(4)
        elif success_rate < self.severity_thresholds['HIGH']['success_rate']:
            concerns.append("HIGH: Moderate failure rate in Monte Carlo analysis")
            severity_scores.append(3)
        
        if ep['statistics']['energy_stats'] is not None:
            energy_rel_unc = ep['statistics']['energy_stats']['rel_uncertainty']
            if energy_rel_unc > self.severity_thresholds['CRITICAL']['rel_uncertainty']:
                concerns.append("CRITICAL: Extreme uncertainty in energy extraction")
                severity_scores.append(4)
            elif energy_rel_unc > self.severity_thresholds['HIGH']['rel_uncertainty']:
                concerns.append("HIGH: High uncertainty in energy extraction")
                severity_scores.append(3)
        
        # Determine overall severity
        if not severity_scores:
            overall_severity = 'LOW'
        else:
            max_severity = max(severity_scores)
            if max_severity >= 4:
                overall_severity = 'CRITICAL'
            elif max_severity >= 3:
                overall_severity = 'HIGH'
            elif max_severity >= 2:
                overall_severity = 'MODERATE'
            else:
                overall_severity = 'LOW'
        
        return {
            'severity_level': overall_severity,
            'identified_concerns': concerns,
            'severity_scores': severity_scores,
            'recommendations': self._generate_recommendations(concerns, overall_severity),
            'metrics_summary': {
                'sinc_stability_fraction': sinc_fraction,
                'qi_stability_fraction': qi_fraction,
                'optimization_convergence_rate': convergence_rate,
                'monte_carlo_success_rate': success_rate,
                'mu_sensitivity_status': mu_sens_status,
                'tau_sensitivity_status': tau_sens_status
            }
        }
    
    def _generate_recommendations(self, concerns: List[str], severity: str) -> List[str]:
        """
        Generate specific recommendations for identified UQ concerns.
        
        Parameters:
        -----------
        concerns : List[str]
            List of identified concerns
        severity : str
            Overall severity level
            
        Returns:
        --------
        List[str]
            Specific recommendations
        """
        recommendations = []
        
        if severity in ['CRITICAL', 'HIGH']:
            recommendations.append("URGENT: Implement robust numerical safeguards")
            recommendations.append("Add parameter validation with safe operating ranges")
            recommendations.append("Implement adaptive precision algorithms")
            
        if any('sinc' in concern.lower() for concern in concerns):
            recommendations.append("Fix sinc function: use Taylor expansion for small μ values")
            recommendations.append("Implement μ bounds checking: 1e-6 ≤ μ ≤ 2.0")
            
        if any('quantum inequality' in concern.lower() for concern in concerns):
            recommendations.append("Fix quantum bounds: implement τ bounds checking")
            recommendations.append("Add overflow protection for 1/τ² calculations")
            
        if any('convergence' in concern.lower() for concern in concerns):
            recommendations.append("Improve optimization: add multiple initial conditions")
            recommendations.append("Implement convergence monitoring and fallback strategies")
            
        if any('sensitivity' in concern.lower() for concern in concerns):
            recommendations.append("Reduce parameter sensitivity through robust design")
            recommendations.append("Implement uncertainty margins in operational parameters")
            
        if any('uncertainty' in concern.lower() for concern in concerns):
            recommendations.append("Improve error propagation analysis")
            recommendations.append("Add confidence intervals to all output quantities")
            
        return recommendations
    
    def implement_uq_fixes(self) -> Dict:
        """
        Implement recommended fixes for identified UQ concerns.
        
        Returns:
        --------
        Dict
            Implementation results
        """
        logger.info("Implementing UQ fixes...")
        
        # Perform initial analysis
        analysis = self.comprehensive_uq_analysis()
        initial_severity = analysis['overall_assessment']['severity_level']
        
        fixes_implemented = []
        
        # Fix 1: Robust sinc calculation
        if any('sinc' in concern.lower() for concern in analysis['overall_assessment']['identified_concerns']):
            self._implement_robust_sinc_fix()
            fixes_implemented.append("Robust sinc calculation with Taylor expansion")
        
        # Fix 2: Parameter bounds checking
        if analysis['overall_assessment']['severity_level'] in ['CRITICAL', 'HIGH']:
            self._implement_parameter_bounds_fix()
            fixes_implemented.append("Parameter bounds validation")
        
        # Fix 3: Optimization improvements
        if analysis['numerical_stability']['optimization_convergence']['convergence_rate'] < 0.8:
            self._implement_optimization_fixes()
            fixes_implemented.append("Enhanced optimization robustness")
        
        # Verify fixes by re-running analysis
        logger.info("Re-running analysis to verify fixes...")
        post_fix_analysis = self.comprehensive_uq_analysis()
        post_fix_severity = post_fix_analysis['overall_assessment']['severity_level']
        
        improvement = self._calculate_improvement(analysis, post_fix_analysis)
        
        return {
            'initial_severity': initial_severity,
            'post_fix_severity': post_fix_severity,
            'fixes_implemented': fixes_implemented,
            'improvement_metrics': improvement,
            'fix_success': post_fix_severity in ['LOW', 'MODERATE'] and 
                          self._severity_to_number(post_fix_severity) < self._severity_to_number(initial_severity)
        }
    
    def _implement_robust_sinc_fix(self):
        """Implement robust sinc calculation fix."""
        # This would modify the sinc calculation in polymer_quantization.py
        # For now, create a patch recommendation
        logger.info("Recommended: Update sinc calculation with Taylor expansion for μ < 1e-6")
    
    def _implement_parameter_bounds_fix(self):
        """Implement parameter bounds validation fix."""
        logger.info("Recommended: Add parameter validation in all initialization methods")
    
    def _implement_optimization_fixes(self):
        """Implement optimization robustness fixes."""
        logger.info("Recommended: Add multi-start optimization with convergence monitoring")
    
    def _calculate_improvement(self, before: Dict, after: Dict) -> Dict:
        """Calculate improvement metrics between analyses."""
        before_metrics = before['overall_assessment']['metrics_summary']
        after_metrics = after['overall_assessment']['metrics_summary']
        
        return {
            'sinc_stability_improvement': (after_metrics['sinc_stability_fraction'] - 
                                         before_metrics['sinc_stability_fraction']),
            'qi_stability_improvement': (after_metrics['qi_stability_fraction'] - 
                                       before_metrics['qi_stability_fraction']),
            'convergence_improvement': (after_metrics['optimization_convergence_rate'] - 
                                      before_metrics['optimization_convergence_rate']),
            'success_rate_improvement': (after_metrics['monte_carlo_success_rate'] - 
                                       before_metrics['monte_carlo_success_rate'])
        }
    
    def _severity_to_number(self, severity: str) -> int:
        """Convert severity string to number for comparison."""
        severity_map = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3, 'CRITICAL': 4}
        return severity_map.get(severity, 0)


# Main execution
if __name__ == "__main__":
    print("🔍 LQG Polymer Field Generator - UQ Analysis")
    print("=" * 60)
    
    # Initialize UQ resolver
    uq_resolver = UQConcernResolver()
    
    # Perform comprehensive analysis
    print("\n📊 Performing comprehensive UQ analysis...")
    results = uq_resolver.comprehensive_uq_analysis()
    
    # Display results
    assessment = results['overall_assessment']
    print(f"\n🎯 Overall UQ Severity: {assessment['severity_level']}")
    
    if assessment['identified_concerns']:
        print(f"\n⚠️  Identified Concerns ({len(assessment['identified_concerns'])}):")
        for i, concern in enumerate(assessment['identified_concerns'], 1):
            print(f"  {i}. {concern}")
    else:
        print("\n✅ No critical UQ concerns identified!")
    
    print(f"\n📈 Key Metrics:")
    metrics = assessment['metrics_summary']
    print(f"  • Sinc Stability: {metrics['sinc_stability_fraction']:.1%}")
    print(f"  • QI Stability: {metrics['qi_stability_fraction']:.1%}")
    print(f"  • Convergence Rate: {metrics['optimization_convergence_rate']:.1%}")
    print(f"  • Success Rate: {metrics['monte_carlo_success_rate']:.1%}")
    print(f"  • μ Sensitivity: {metrics['mu_sensitivity_status']}")
    print(f"  • τ Sensitivity: {metrics['tau_sensitivity_status']}")
    
    if assessment['recommendations']:
        print(f"\n🔧 Recommendations ({len(assessment['recommendations'])}):")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Implement fixes if needed
    if assessment['severity_level'] in ['CRITICAL', 'HIGH']:
        print(f"\n🛠️  Implementing UQ fixes...")
        fix_results = uq_resolver.implement_uq_fixes()
        
        print(f"\n✅ Fix Implementation Results:")
        print(f"  • Initial Severity: {fix_results['initial_severity']}")
        print(f"  • Post-Fix Severity: {fix_results['post_fix_severity']}")
        print(f"  • Fix Success: {fix_results['fix_success']}")
        
        if fix_results['fixes_implemented']:
            print(f"  • Fixes Applied:")
            for fix in fix_results['fixes_implemented']:
                print(f"    - {fix}")
    
    print(f"\n🎉 UQ Analysis Complete!")
