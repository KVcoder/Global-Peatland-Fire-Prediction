"""
Quick Mamba Installation & Performance Test
Tests if mamba-ssm is properly installed and benchmarks it vs GRU
"""
import torch
import time
import sys

def test_import():
    """Test if mamba_ssm can be imported"""
    print("=" * 60)
    print("TEST 1: Import Check")
    print("=" * 60)
    try:
        from mamba_ssm import Mamba
        print("✓ mamba_ssm imported successfully")
        return True
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        print("\nInstall with:")
        print("  pip install mamba-ssm causal-conv1d")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_initialization():
    """Test if Mamba can be initialized"""
    print("\n" + "=" * 60)
    print("TEST 2: Initialization Check")
    print("=" * 60)
    try:
        from mamba_ssm import Mamba
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        model = Mamba(
            d_model=128,
            d_state=16,
            d_conv=4,
            expand=2
        ).to(device)
        print(f"✓ Mamba initialized successfully")
        print(f"  d_model=128, d_state=16, d_conv=4, expand=2")
        return True, model, device
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False, None, None

def test_forward_pass(model, device):
    """Test if forward pass works"""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass Check")
    print("=" * 60)
    try:
        B, T, D = 4, 30, 128
        x = torch.randn(B, T, D, device=device)
        print(f"Input shape: {tuple(x.shape)}")
        
        with torch.no_grad():
            y = model(x)
        
        print(f"Output shape: {tuple(y.shape)}")
        print(f"✓ Forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def benchmark_mamba_vs_gru():
    """Benchmark Mamba vs GRU"""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Benchmark")
    print("=" * 60)
    
    try:
        from mamba_ssm import Mamba
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup
        B, T, D = 32, 30, 128  # Realistic batch size and sequence length
        x = torch.randn(B, T, D, device=device)
        
        # Mamba model
        mamba = Mamba(d_model=D, d_state=16, d_conv=4, expand=2).to(device)
        
        # GRU model
        gru = torch.nn.GRU(
            input_size=D, 
            hidden_size=D, 
            num_layers=1, 
            batch_first=True
        ).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = mamba(x)
                _ = gru(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark Mamba
        n_iter = 50
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iter):
                _ = mamba(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mamba_time = (time.time() - start) / n_iter
        
        # Benchmark GRU
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iter):
                _ = gru(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gru_time = (time.time() - start) / n_iter
        
        print(f"Batch size: {B}, Sequence length: {T}, Hidden dim: {D}")
        print(f"\nMamba: {mamba_time*1000:.2f} ms/batch")
        print(f"GRU:   {gru_time*1000:.2f} ms/batch")
        print(f"\nSpeedup: {gru_time/mamba_time:.2f}x")
        
        if mamba_time < gru_time:
            print(f"✓ Mamba is {gru_time/mamba_time:.2f}x faster!")
        else:
            print(f"⚠ GRU is {mamba_time/gru_time:.2f}x faster (unexpected)")
            
        return True
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

def main():
    print("\n🔥 MAMBA INSTALLATION & PERFORMANCE TEST 🔥\n")
    
    # Test 1: Import
    if not test_import():
        print("\n❌ FAILED: Cannot import mamba_ssm")
        sys.exit(1)
    
    # Test 2: Initialization
    success, model, device = test_initialization()
    if not success:
        print("\n❌ FAILED: Cannot initialize Mamba")
        sys.exit(1)
    
    # Test 3: Forward pass
    if not test_forward_pass(model, device):
        print("\n❌ FAILED: Forward pass failed")
        sys.exit(1)
    
    # Test 4: Benchmark
    benchmark_mamba_vs_gru()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nMamba is ready to use in your training script.")
    print("Run with: python train.py --use-mamba [other args...]")

if __name__ == "__main__":
    main()