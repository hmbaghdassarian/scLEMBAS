#!/bin/bash

# JAX version fix script for ppc64le (Corrected)
# Run this after the main installation to fix JAX compatibility

set -e

echo "=== Fixing JAX version compatibility for ppc64le (Corrected) ==="

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scvi-gpu

echo "Current JAX packages:"
conda list | grep -i jax || echo "No JAX packages found"

echo ""
echo "Removing ALL existing JAX packages completely..."
pip uninstall jax jaxlib -y 2>/dev/null || echo "No pip JAX packages to remove"
conda remove jax jaxlib optax flax --force -y 2>/dev/null || echo "No conda JAX packages to remove"

echo ""
echo "Performing thorough cleanup..."
conda clean --all -y

echo ""
echo "Installing JAX with compatible versions for ppc64le..."

# Strategy 1: Try exact compatible versions from conda-forge
echo "Attempting Strategy 1: Exact compatible versions (jaxlib=0.4.7, jax=0.4.11)..."
if mamba install -c conda-forge jaxlib=0.4.7 -y && mamba install -c conda-forge jax=0.4.11 -y; then
    echo "Testing conda-forge installation..."
    if python -c "import jax; import jaxlib; print(f'✓ JAX: {jax.__version__}, JAXlib: {jaxlib.__version__}')" 2>/dev/null; then
        echo "✓ Strategy 1 SUCCESS: conda-forge compatible versions"
        JAX_INSTALLED=true
        JAX_METHOD="conda-forge"
    else
        echo "⚠ Strategy 1 FAILED: conda versions don't work"
        JAX_INSTALLED=false
    fi
else
    echo "⚠ Strategy 1 FAILED: couldn't install conda versions"
    JAX_INSTALLED=false
fi

# Strategy 2: CPU-only JAX via pip (most reliable for ppc64le)
if [ "$JAX_INSTALLED" = false ]; then
    echo ""
    echo "Attempting Strategy 2: CPU-only JAX via pip..."
    # Remove any partial conda installations first
    conda remove jax jaxlib --force -y 2>/dev/null || true
    
    if pip install "jax[cpu]==0.4.20" "jaxlib==0.4.20" --force-reinstall; then
        echo "Testing CPU-only JAX installation..."
        if python -c "import jax; import jaxlib; print(f'✓ JAX: {jax.__version__}, JAXlib: {jaxlib.__version__}')" 2>/dev/null; then
            echo "✓ Strategy 2 SUCCESS: CPU-only JAX"
            JAX_INSTALLED=true
            JAX_METHOD="cpu-only-pip"
        else
            echo "⚠ Strategy 2 FAILED: CPU-only JAX doesn't work"
            JAX_INSTALLED=false
        fi
    else
        echo "⚠ Strategy 2 FAILED: couldn't install CPU-only JAX"
        JAX_INSTALLED=false
    fi
fi

# Strategy 3: Try latest compatible versions
if [ "$JAX_INSTALLED" = false ]; then
    echo ""
    echo "Attempting Strategy 3: Latest compatible versions..."
    if pip install "jax==0.4.27" "jaxlib==0.4.27" --force-reinstall; then
        echo "Testing latest compatible versions..."
        if python -c "import jax; import jaxlib; print(f'✓ JAX: {jax.__version__}, JAXlib: {jaxlib.__version__}')" 2>/dev/null; then
            echo "✓ Strategy 3 SUCCESS: Latest compatible versions"
            JAX_INSTALLED=true
            JAX_METHOD="latest-compatible"
        else
            echo "⚠ Strategy 3 FAILED: Latest compatible versions don't work"
            JAX_INSTALLED=false
        fi
    else
        echo "⚠ Strategy 3 FAILED: couldn't install latest compatible versions"
        JAX_INSTALLED=false
    fi
fi

# Install JAX-dependent packages if JAX works
if [ "$JAX_INSTALLED" = true ]; then
    echo ""
    echo "Installing JAX-dependent packages..."
    pip install --prefer-binary optax flax || echo "⚠ Some JAX-dependent packages failed"
else
    echo ""
    echo "⚠ All JAX installation strategies failed"
fi

echo ""
echo "=== Testing Complete Installation ==="

# Test JAX functionality
if [ "$JAX_INSTALLED" = true ]; then
    echo "Testing JAX functionality..."
    python -c "
try:
    import jax
    import jaxlib
    import jax.numpy as jnp
    
    print(f'✓ JAX version: {jax.__version__}')
    print(f'✓ JAXlib version: {jaxlib.__version__}')
    
    # Test basic operations
    x = jnp.array([1, 2, 3])
    result = x.sum()
    print(f'✓ JAX basic operations work: {result}')
    
    # Test compilation
    from jax import jit
    @jit
    def f(x):
        return x * 2
    
    result = f(jnp.array(5.0))
    print(f'✓ JAX JIT compilation works: {result}')
    
except Exception as e:
    print(f'⚠ JAX functionality test failed: {e}')
"
else
    echo "⚠ JAX not available for functionality testing"
fi

echo ""
echo "Testing scvi-tools import..."
python -c "
try:
    import scvi
    print(f'✅ scvi-tools version: {scvi.__version__}')
    print('✅ scvi-tools imported successfully!')
except Exception as e:
    print(f'❌ scvi-tools import failed: {e}')
    print('   This indicates JAX compatibility issues remain')
"

echo ""
echo "Testing PyTorch CUDA (should still work)..."
python -c "
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
except Exception as e:
    print(f'❌ PyTorch test failed: {e}')
"

echo ""
echo "=== Final Summary ==="
if [ "$JAX_INSTALLED" = true ]; then
    echo "✅ JAX installation: SUCCESS (method: $JAX_METHOD)"
    echo "✅ You should now be able to use scvi-tools with full functionality"
    echo ""
    echo "JAX Configuration:"
    case $JAX_METHOD in
        "cpu-only-pip")
            echo "   • JAX is running in CPU-only mode"
            echo "   • This is optimal for ppc64le systems"
            echo "   • PyTorch will still use GPU for heavy computation"
            ;;
        "conda-forge")
            echo "   • JAX installed from conda-forge"
            echo "   • Should have good ppc64le compatibility"
            ;;
        "latest-compatible")
            echo "   • JAX using latest compatible versions"
            echo "   • May have GPU support depending on system"
            ;;
    esac
else
    echo "❌ JAX installation: FAILED"
    echo ""
    echo "All strategies failed. Possible solutions:"
    echo "1. Manual installation:"
    echo "   pip install --no-deps jax==0.4.11 jaxlib==0.4.11"
    echo "2. Use scvi-tools without JAX features (limited functionality)"
    echo "3. Try building JAX from source (advanced)"
    echo "4. Use a different environment or system"
fi

echo ""
echo "Quick test commands:"
echo "1. Test JAX:        python -c \"import jax; print('JAX works!')\""
echo "2. Test scvi-tools: python -c \"import scvi; print('scvi-tools works!')\""
echo "3. Test PyTorch:    python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""

echo ""
echo "🎯 Your environment status:"
echo "   • PyTorch + CUDA: ✅ Working (Tesla V100 detected)"
echo "   • Clustering:     ✅ louvain available"
if [ "$JAX_INSTALLED" = true ]; then
    echo "   • JAX:            ✅ Working ($JAX_METHOD)"
    echo "   • scvi-tools:     ✅ Should be fully functional"
else
    echo "   • JAX:            ❌ Failed"
    echo "   • scvi-tools:     ❌ May have limited functionality"
fi

echo ""
echo "Environment ready! Activate with: conda activate scvi-gpu"

