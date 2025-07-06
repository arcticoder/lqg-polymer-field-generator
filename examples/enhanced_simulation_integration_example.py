#!/usr/bin/env python3
"""
LQG Polymer Field Generator - Enhanced Simulation Integration Example

This example demonstrates the deep integration between the LQG Polymer Field Generator
and the Enhanced Simulation Hardware Abstraction Framework, showcasing:

- Hardware-abstracted polymer field generation
- Real-time digital twin synchronization 
- Multi-physics coupling with LQG corrections
- Enhanced metamaterial amplification
- Comprehensive cross-system UQ analysis

Author: LQG-FTL Research Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from integration.enhanced_simulation_integration import (
        create_lqg_enhanced_simulation_integration,
        LQGEnhancedSimulationConfig
    )
except ImportError:
    # Fallback for development
    sys.path.append(str(Path(__file__).parent))
    from enhanced_simulation_integration import (
        create_lqg_enhanced_simulation_integration,
        LQGEnhancedSimulationConfig
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_basic_integration_example():
    """Run basic integration example with default parameters"""
    
    logger.info("=== LQG Polymer Field Generator - Enhanced Simulation Integration Example ===")
    
    # Create integration with default configuration
    integration = create_lqg_enhanced_simulation_integration()
    
    # Define spatial and temporal domains
    spatial_domain = np.linspace(-5, 5, 200)  # 200 spatial points
    temporal_domain = np.linspace(0, 10, 100)  # 100 temporal points
    
    logger.info(f"Spatial domain: {len(spatial_domain)} points from {spatial_domain[0]:.1f} to {spatial_domain[-1]:.1f}")
    logger.info(f"Temporal domain: {len(temporal_domain)} points from {temporal_domain[0]:.1f} to {temporal_domain[-1]:.1f}")
    
    # Run integrated field generation
    logger.info("Starting integrated polymer field generation...")
    results = integration.generate_polymer_field_with_hardware_abstraction(spatial_domain, temporal_domain)
    
    # Extract key results
    final_field = results['final_field']
    integration_metrics = results['integration_metrics']
    validation_status = results['validation_status']
    uq_analysis = results['uq_analysis']
    
    # Display results summary
    print("\n" + "="*60)
    print("INTEGRATION RESULTS SUMMARY")
    print("="*60)
    
    print(f"Processing time: {results['processing_time']:.3f} seconds")
    print(f"Validation status: {validation_status['overall_status']}")
    print(f"Validation score: {validation_status['validation_score']:.2f}")
    print(f"Integration score: {integration_metrics['integration_score']:.2f}")
    
    # Enhancement factors
    print(f"\nENHANCEMENT FACTORS:")
    print(f"  Base polymer enhancement: {final_field.get('enhancement_factor', 1.0):.2e}")
    if final_field.get('metamaterial_amplification'):
        print(f"  Metamaterial amplification: {final_field['metamaterial_amplification']:.2e}")
    print(f"  Total enhancement: {integration_metrics['total_enhancement_factor']:.2e}")
    
    # Integration success
    print(f"\nINTEGRATION SUCCESS:")
    for component, success in integration_metrics['integration_success'].items():
        status = "✓" if success else "✗"
        print(f"  {status} {component.replace('_', ' ').title()}")
    
    # UQ analysis results
    if uq_analysis.get('uq_analysis_complete'):
        print(f"\nUNCERTAINTY QUANTIFICATION:")
        print(f"  Overall confidence: {uq_analysis['overall_confidence']:.2f}")
        print(f"  Total uncertainty: {uq_analysis['integration_uncertainty']['total_uncertainty']:.4f}")
        
        field_stats = uq_analysis['field_statistics']
        print(f"  Field SNR: {field_stats['snr']:.2e}")
        print(f"  Field range: [{field_stats['amplitude_range'][0]:.2e}, {field_stats['amplitude_range'][1]:.2e}]")
    
    # Target achievement
    targets = integration_metrics['target_achievement']
    print(f"\nTARGET ACHIEVEMENT:")
    for target, achieved in targets.items():
        status = "✓" if achieved else "✗"
        print(f"  {status} {target.replace('_', ' ').title()}")
    
    return results

def run_advanced_integration_example():
    """Run advanced integration example with custom configuration"""
    
    logger.info("=== Advanced Integration Example ===")
    
    # Create custom configuration
    config = LQGEnhancedSimulationConfig(
        polymer_parameter_mu=0.8,  # Higher polymer parameter
        field_resolution=500,  # Higher spatial resolution
        temporal_steps=200,  # More temporal steps
        target_precision=0.05e-12,  # Tighter precision target
        target_amplification=2.0e10,  # Higher amplification target
        monte_carlo_samples=5000,  # More UQ samples
        enable_real_time_monitoring=True
    )
    
    # Create integration with custom config
    integration = create_lqg_enhanced_simulation_integration(config)
    
    # Define high-resolution domains
    spatial_domain = np.linspace(-8, 8, config.field_resolution)
    temporal_domain = np.linspace(0, 15, config.temporal_steps)
    
    logger.info(f"High-resolution configuration:")
    logger.info(f"  Polymer parameter μ: {config.polymer_parameter_mu}")
    logger.info(f"  Spatial resolution: {config.field_resolution}")
    logger.info(f"  Temporal steps: {config.temporal_steps}")
    logger.info(f"  Precision target: {config.target_precision:.2e} m/√Hz")
    logger.info(f"  Amplification target: {config.target_amplification:.2e}×")
    
    # Run integration
    results = integration.generate_polymer_field_with_hardware_abstraction(spatial_domain, temporal_domain)
    
    # Advanced analysis
    analyze_integration_performance(results, config)
    
    return results

def analyze_integration_performance(results, config):
    """Analyze integration performance in detail"""
    
    print("\n" + "="*60)
    print("ADVANCED PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Component-wise analysis
    components = ['base_field', 'enhanced_field', 'synchronized_field', 'amplified_field', 'measured_field', 'final_field']
    
    print(f"\nCOMPONENT-WISE ENHANCEMENT FACTORS:")
    base_enhancement = results['base_field']['enhancement_factor']
    
    for component in components:
        if component in results:
            field_data = results[component]
            current_enhancement = field_data.get('enhancement_factor', base_enhancement)
            
            # Calculate relative enhancement from previous stage
            if component == 'base_field':
                relative_enhancement = 1.0
            else:
                relative_enhancement = current_enhancement / base_enhancement
            
            print(f"  {component.replace('_', ' ').title()}: {current_enhancement:.2e} ({relative_enhancement:.2f}× relative)")
    
    # UQ uncertainty breakdown
    if results['uq_analysis'].get('uq_analysis_complete'):
        print(f"\nUNCERTAINTY BREAKDOWN:")
        uncertainty = results['uq_analysis']['integration_uncertainty']
        
        uncertainty_sources = [
            ('Polymer Field', uncertainty['polymer_uncertainty']),
            ('Hardware Abstraction', uncertainty['hardware_uncertainty']),
            ('Digital Twin Sync', uncertainty['sync_uncertainty']),
            ('Metamaterial Amplification', uncertainty['metamaterial_uncertainty']),
            ('Precision Measurement', uncertainty['measurement_uncertainty']),
            ('Multi-Physics Coupling', uncertainty['coupling_uncertainty'])
        ]
        
        for source, value in uncertainty_sources:
            percentage = (value / uncertainty['total_uncertainty']) * 100 if uncertainty['total_uncertainty'] > 0 else 0
            print(f"  {source}: {value:.4f} ({percentage:.1f}%)")
        
        print(f"  Total Combined: {uncertainty['total_uncertainty']:.4f}")
    
    # Performance vs targets
    print(f"\nPERFORMANCE VS TARGETS:")
    integration_metrics = results['integration_metrics']
    target_achievement = integration_metrics['target_achievement']
    
    final_field = results['final_field']
    
    # Precision achievement
    actual_precision = final_field.get('measurement_precision', 1e-12)
    precision_ratio = actual_precision / config.target_precision
    print(f"  Precision: {actual_precision:.2e} / {config.target_precision:.2e} = {precision_ratio:.2f}× target")
    
    # Amplification achievement  
    actual_amplification = final_field.get('metamaterial_amplification', 1.0)
    amplification_ratio = actual_amplification / config.target_amplification
    print(f"  Amplification: {actual_amplification:.2e} / {config.target_amplification:.2e} = {amplification_ratio:.2f}× target")
    
    # Fidelity achievement
    actual_fidelity = final_field.get('digital_twin_fidelity', 0.0)
    fidelity_ratio = actual_fidelity / config.target_fidelity
    print(f"  Fidelity: {actual_fidelity:.3f} / {config.target_fidelity:.3f} = {fidelity_ratio:.2f}× target")

def plot_integration_results(results):
    """Plot integration results for visualization"""
    
    try:
        # Extract field data
        base_field = results['base_field']
        final_field = results['final_field']
        
        spatial_domain = base_field['spatial_domain']
        temporal_domain = base_field['temporal_domain']
        base_amplitude = base_field['field_amplitude']
        final_amplitude = final_field['field_amplitude']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('LQG Polymer Field Generator - Enhanced Simulation Integration Results', fontsize=14)
        
        # Plot 1: Base polymer field (spatial profile at t=0)
        axes[0, 0].plot(spatial_domain, base_amplitude[:, 0], 'b-', linewidth=2, label='Base polymer field')
        axes[0, 0].set_xlabel('Spatial coordinate')
        axes[0, 0].set_ylabel('Field amplitude')
        axes[0, 0].set_title('Base Polymer Field (t=0)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Final integrated field (spatial profile at t=0)
        axes[0, 1].plot(spatial_domain, final_amplitude[:, 0], 'r-', linewidth=2, label='Integrated field')
        axes[0, 1].set_xlabel('Spatial coordinate')
        axes[0, 1].set_ylabel('Field amplitude')
        axes[0, 1].set_title('Final Integrated Field (t=0)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Temporal evolution comparison (at x=0)
        center_idx = len(spatial_domain) // 2
        axes[1, 0].plot(temporal_domain, base_amplitude[center_idx, :], 'b-', linewidth=2, label='Base field')
        axes[1, 0].plot(temporal_domain, final_amplitude[center_idx, :], 'r-', linewidth=2, label='Integrated field')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Field amplitude')
        axes[1, 0].set_title('Temporal Evolution (x=0)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Enhancement factor comparison
        enhancement_stages = ['Base', 'Enhanced', 'Synchronized', 'Amplified', 'Measured', 'Final']
        enhancement_values = []
        
        base_enhancement = base_field['enhancement_factor']
        enhancement_values.append(base_enhancement)
        
        components = ['enhanced_field', 'synchronized_field', 'amplified_field', 'measured_field', 'final_field']
        for component in components:
            if component in results:
                field_data = results[component]
                current_enhancement = field_data.get('enhancement_factor', base_enhancement)
                enhancement_values.append(current_enhancement)
            else:
                enhancement_values.append(base_enhancement)
        
        axes[1, 1].semilogy(enhancement_stages, enhancement_values, 'go-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Integration Stage')
        axes[1, 1].set_ylabel('Enhancement Factor')
        axes[1, 1].set_title('Enhancement Factor by Stage')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("lqg_enhanced_simulation_results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "integration_results.png", dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_dir / 'integration_results.png'}")
        
        plt.show()
        
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

def run_uq_analysis_example():
    """Run detailed UQ analysis example"""
    
    logger.info("=== UQ Analysis Example ===")
    
    # Create configuration with enhanced UQ
    config = LQGEnhancedSimulationConfig(
        monte_carlo_samples=10000,  # High sampling for detailed UQ
        uq_confidence_level=0.99,  # 99% confidence
        enable_cross_system_uq=True,
        enable_real_time_monitoring=True
    )
    
    integration = create_lqg_enhanced_simulation_integration(config)
    
    # Run multiple realizations for statistical analysis
    n_realizations = 5
    all_results = []
    
    logger.info(f"Running {n_realizations} realizations for statistical analysis...")
    
    spatial_domain = np.linspace(-3, 3, 100)
    temporal_domain = np.linspace(0, 5, 50)
    
    for i in range(n_realizations):
        logger.info(f"Realization {i+1}/{n_realizations}")
        results = integration.generate_polymer_field_with_hardware_abstraction(spatial_domain, temporal_domain)
        all_results.append(results)
    
    # Analyze statistical properties across realizations
    analyze_statistical_properties(all_results)
    
    return all_results

def analyze_statistical_properties(all_results):
    """Analyze statistical properties across multiple realizations"""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS ACROSS REALIZATIONS")
    print("="*60)
    
    # Extract key metrics from all realizations
    enhancement_factors = []
    integration_scores = []
    confidence_levels = []
    total_uncertainties = []
    validation_scores = []
    
    for results in all_results:
        # Enhancement factors
        enhancement_factors.append(results['integration_metrics']['total_enhancement_factor'])
        
        # Integration scores  
        integration_scores.append(results['integration_metrics']['integration_score'])
        
        # UQ metrics
        if results['uq_analysis'].get('uq_analysis_complete'):
            confidence_levels.append(results['uq_analysis']['overall_confidence'])
            total_uncertainties.append(results['uq_analysis']['integration_uncertainty']['total_uncertainty'])
        
        # Validation scores
        validation_scores.append(results['validation_status']['validation_score'])
    
    # Statistical summary
    print(f"NUMBER OF REALIZATIONS: {len(all_results)}")
    
    if enhancement_factors:
        print(f"\nENHANCEMENT FACTORS:")
        print(f"  Mean: {np.mean(enhancement_factors):.2e}")
        print(f"  Std:  {np.std(enhancement_factors):.2e}")
        print(f"  Min:  {np.min(enhancement_factors):.2e}")
        print(f"  Max:  {np.max(enhancement_factors):.2e}")
    
    if integration_scores:
        print(f"\nINTEGRATION SCORES:")
        print(f"  Mean: {np.mean(integration_scores):.3f}")
        print(f"  Std:  {np.std(integration_scores):.3f}")
        print(f"  Min:  {np.min(integration_scores):.3f}")
        print(f"  Max:  {np.max(integration_scores):.3f}")
    
    if confidence_levels:
        print(f"\nCONFIDENCE LEVELS:")
        print(f"  Mean: {np.mean(confidence_levels):.3f}")
        print(f"  Std:  {np.std(confidence_levels):.3f}")
        print(f"  Min:  {np.min(confidence_levels):.3f}")
        print(f"  Max:  {np.max(confidence_levels):.3f}")
    
    if total_uncertainties:
        print(f"\nTOTAL UNCERTAINTIES:")
        print(f"  Mean: {np.mean(total_uncertainties):.4f}")
        print(f"  Std:  {np.std(total_uncertainties):.4f}")
        print(f"  Min:  {np.min(total_uncertainties):.4f}")
        print(f"  Max:  {np.max(total_uncertainties):.4f}")
    
    if validation_scores:
        print(f"\nVALIDATION SCORES:")
        print(f"  Mean: {np.mean(validation_scores):.3f}")
        print(f"  Std:  {np.std(validation_scores):.3f}")
        print(f"  Success rate: {sum(1 for score in validation_scores if score >= 0.8) / len(validation_scores):.1%}")

def main():
    """Main function demonstrating all integration examples"""
    
    print("LQG Polymer Field Generator - Enhanced Simulation Integration Examples")
    print("====================================================================")
    
    try:
        # Run basic integration example
        print("\n1. Running Basic Integration Example...")
        basic_results = run_basic_integration_example()
        
        # Plot results
        print("\n2. Generating Visualization...")
        plot_integration_results(basic_results)
        
        # Run advanced integration example
        print("\n3. Running Advanced Integration Example...")
        advanced_results = run_advanced_integration_example()
        
        # Run UQ analysis example
        print("\n4. Running UQ Analysis Example...")
        uq_results = run_uq_analysis_example()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: lqg_enhanced_simulation_results/")
        
        return {
            'basic_results': basic_results,
            'advanced_results': advanced_results,
            'uq_results': uq_results
        }
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
