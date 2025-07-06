#!/usr/bin/env python3
"""
Basic Field Generation Example
============================

This example demonstrates the basic usage of the LQG Polymer Field Generator
for generating sinc(πμ) enhancement fields using quantum geometric field manipulation.

This is the simplest way to get started with the field generator and validates
that the UQ framework is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Basic field generation demonstration."""
    print("LQG Polymer Field Generator - Basic Example")
    print("=" * 50)
    
    try:
        # Import core components
        from core.polymer_quantization import PolymerFieldGenerator
        from optimization.parameter_selection import OptimalParameters
        from optimization.robust_optimizer import RobustParameterValidator
        from validation.uq_analysis import UQAnalysisFramework
        
        print("✅ All imports successful")
        
        # Initialize the generator with robust validation
        print("\n1. Initializing LQG Polymer Field Generator...")
        generator = PolymerFieldGenerator()
        validator = RobustParameterValidator()
        print("✅ Generator initialized")
        
        # Set optimal parameters with validation
        print("\n2. Setting optimal parameters...")
        params = OptimalParameters()
        validated_mu = validator.validate_mu_parameter(params.mu_optimal)
        generator.configure(mu=validated_mu)
        print(f"✅ Parameters configured: μ = {validated_mu}")
        
        # Generate enhancement field
        print("\n3. Generating sinc(πμ) enhancement field...")
        field = generator.generate_sinc_enhancement_field()
        print(f"✅ Field generated with enhancement factor: {field.enhancement_factor:.6f}")
        
        # Run UQ analysis to validate the system
        print("\n4. Running UQ analysis...")
        uq_framework = UQAnalysisFramework()
        uq_results = uq_framework.run_comprehensive_analysis(generator)
        
        print(f"✅ UQ Analysis Complete:")
        print(f"   Status: {uq_results['overall_status']}")
        print(f"   Convergence Rate: {uq_results['convergence_rate']:.1%}")
        print(f"   Numerical Stability: {uq_results['numerical_stability']}")
        
        # Generate spatial field profile for visualization
        print("\n5. Generating spatial field profile...")
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Simple spatial profile (this would use the actual field generation code)
        r = np.sqrt(X**2 + Y**2)
        field_profile = field.enhancement_factor * np.exp(-r**2/4) * np.sinc(np.pi * validated_mu * r)
        
        # Create visualization
        print("\n6. Creating visualization...")
        plt.figure(figsize=(12, 5))
        
        # Field profile plot
        plt.subplot(1, 2, 1)
        contour = plt.contourf(X, Y, field_profile, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Enhancement Field')
        plt.title('LQG Polymer Enhancement Field\nsinc(πμ) Spatial Profile')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Enhancement factor vs mu plot
        plt.subplot(1, 2, 2)
        mu_range = np.linspace(0.1, 2.0, 100)
        enhancement_factors = [np.sinc(np.pi * mu) for mu in mu_range]
        plt.plot(mu_range, enhancement_factors, 'b-', linewidth=2)
        plt.axvline(validated_mu, color='r', linestyle='--', label=f'Optimal μ = {validated_mu}')
        plt.xlabel('μ parameter')
        plt.ylabel('sinc(πμ) Enhancement Factor')
        plt.title('Enhancement Factor vs μ Parameter')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(__file__).parent / "basic_field_generation_output.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
        # Display summary
        print("\n" + "=" * 50)
        print("BASIC FIELD GENERATION COMPLETE")
        print("=" * 50)
        print(f"Enhancement Factor: {field.enhancement_factor:.6f}")
        print(f"Optimal μ Parameter: {validated_mu}")
        print(f"UQ Status: {uq_results['overall_status']}")
        print(f"Convergence Rate: {uq_results['convergence_rate']:.1%}")
        print("✅ System ready for advanced applications")
        
        # Show the plot
        plt.show()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you've installed all dependencies: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error during field generation: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Basic field generation example completed successfully!")
    else:
        print("\n💥 Example failed. Check the error messages above.")
        sys.exit(1)
