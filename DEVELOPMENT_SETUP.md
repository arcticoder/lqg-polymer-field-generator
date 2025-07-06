# Development Setup Guide

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/asciimath/lqg-polymer-field-generator.git
   cd lqg-polymer-field-generator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import numpy, scipy, matplotlib; print('Dependencies installed successfully')"
   ```

## Development Environment

### Virtual Environment (Recommended)
```bash
python -m venv lqg-venv
source lqg-venv/bin/activate  # On Windows: lqg-venv\Scripts\activate
pip install -r requirements.txt
```

### Development Dependencies
```bash
pip install -r requirements.txt
pre-commit install  # Set up git hooks
```

## Project Structure

```
lqg-polymer-field-generator/
├── src/                          # Source code
│   ├── core/                     # Core polymer quantization
│   ├── optimization/             # Robust optimization framework
│   ├── validation/               # UQ analysis framework
│   ├── field_generation/         # Field generation modules
│   └── lagrangian/              # Lagrangian formulation
├── tests/                        # Test suite
├── examples/                     # Usage examples
├── docs/                         # Documentation
├── .github/                      # GitHub metadata
└── UQ_RESOLUTION_SUMMARY.md     # UQ concern resolution
```

## Module Implementation Status

**Implementation Priority**: Based on UQ resolution framework requirements

### ✅ Completed (UQ Resolution Phase)
- UQ analysis framework design
- Robust optimization specifications
- Parameter validation requirements
- Numerical stability improvements
- Documentation framework

### 🔄 Implementation Needed
- Core polymer quantization modules
- Robust optimization implementations
- Field generation components
- Lagrangian formulation
- Integration interfaces

See [MODULE_STRUCTURE_TEMPLATE.md](MODULE_STRUCTURE_TEMPLATE.md) for detailed implementation specifications.

## UQ Framework Requirements

### Critical Requirements (Must Maintain)
- **Convergence Rate**: 100% (improved from 0% during UQ resolution)
- **UQ Status**: HIGH (all critical concerns resolved)
- **Numerical Stability**: No division-by-zero or overflow errors
- **Parameter Validation**: Robust bounds checking with quantum constraints

### UQ Analysis Integration
All new code must integrate with:
```python
from src.validation.uq_analysis import UQAnalysisFramework

uq_framework = UQAnalysisFramework()
uq_results = uq_framework.run_comprehensive_analysis(component)
assert uq_results['overall_status'] == 'HIGH'
assert uq_results['convergence_rate'] >= 1.0
```

## Testing Framework

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### UQ Validation Tests
```bash
pytest tests/uq/ -v
```

### Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for critical optimization paths
- All UQ-related code must have comprehensive tests

## Code Quality Standards

### Formatting
```bash
black src/ tests/
```

### Linting
```bash
flake8 src/ tests/
```

### Type Checking
```bash
mypy src/
```

### Pre-commit Hooks
Automatically run before each commit:
- Black formatting
- Flake8 linting
- Basic UQ validation
- Import sorting

## Development Workflow

### Feature Development
1. Create feature branch from main
2. Implement module following template specifications
3. Add comprehensive tests including UQ validation
4. Update documentation
5. Run full test suite
6. Submit pull request

### UQ Validation Workflow
1. Implement new functionality
2. Run UQ analysis: `python -m src.validation.uq_analysis`
3. Verify UQ status remains HIGH
4. Document any new UQ considerations
5. Update UQ tracking files if needed

## Integration with LQG Ecosystem

### Repository Dependencies
- `unified-lqg`: Core mathematical framework
- `warp-bubble-optimizer`: Bobrick-Martire configurations  
- `negative-energy-generator`: Field algebra operations
- `warp-spacetime-stability-controller`: Real-time validation

### API Compatibility
Maintain standard interfaces for cross-repository integration:
```python
# Standard pattern used across all LQG repositories
generator = PolymerFieldGenerator()
validator = RobustParameterValidator()
field = generator.generate_sinc_enhancement_field()
```

## Documentation Standards

### Code Documentation
- Comprehensive docstrings for all classes and methods
- Mathematical derivations in comments
- Physical interpretation explanations
- Usage examples in docstrings

### Technical Documentation
- Update [docs/technical-documentation.md](docs/technical-documentation.md) for major changes
- Maintain [UQ_RESOLUTION_SUMMARY.md](UQ_RESOLUTION_SUMMARY.md) for UQ-related updates
- Cross-reference with related repositories

## Debugging and Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### UQ Analysis Failures
```bash
# Run detailed UQ analysis
python -m src.validation.uq_analysis --verbose --detailed
```

#### Convergence Issues
```bash
# Test with robust optimizer
python -c "from src.optimization.robust_optimizer import MultiStartOptimizer; print('Robust optimization available')"
```

### Performance Debugging
```bash
# Profile optimization performance
python -m cProfile -o profile.stats examples/basic_field_generation.py
```

## Contribution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines including:
- Code review process
- UQ validation requirements
- Integration testing procedures
- Documentation standards

## Security Considerations

See [SECURITY.md](SECURITY.md) for:
- Vulnerability reporting procedures
- UQ security considerations
- Safe usage guidelines
- Parameter validation security

---

**Ready to Contribute?**

1. Fork the repository
2. Set up your development environment
3. Choose a module from the template to implement
4. Follow the UQ validation requirements
5. Submit a pull request

For questions, open an issue or start a discussion!
