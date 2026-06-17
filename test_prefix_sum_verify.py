import torch
import bh_ops

def test_prefix_sum():
    device = "cuda"

    # Test 1: Small input
    x = torch.randn(100, device=device)
    out = bh_ops.scan_then_fan(x)
    ref = torch.cumsum(x, dim=0)
    print(f"Test1 (n=100):  max_err={torch.max(torch.abs(out - ref)).item():.2e}, PASS={torch.allclose(out, ref, atol=1e-4)}")

    # Test 2: Exact perBlockNum boundary
    x = torch.randn(1024, device=device)
    out = bh_ops.scan_then_fan(x)
    ref = torch.cumsum(x, dim=0)
    print(f"Test2 (n=1024): max_err={torch.max(torch.abs(out - ref)).item():.2e}, PASS={torch.allclose(out, ref, atol=1e-4)}")

    # Test 3: Larger than one block
    x = torch.randn(5000, device=device)
    out = bh_ops.scan_then_fan(x)
    ref = torch.cumsum(x, dim=0)
    print(f"Test3 (n=5000): max_err={torch.max(torch.abs(out - ref)).item():.2e}, PASS={torch.allclose(out, ref, atol=1e-4)}")

    # Test 4: Very large
    x = torch.randn(1000000, device=device)
    out = bh_ops.scan_then_fan(x)
    ref = torch.cumsum(x, dim=0)
    print(f"Test4 (n=1M):   max_err={torch.max(torch.abs(out - ref)).item():.2e}, PASS={torch.allclose(out, ref, atol=1e-3)}")

    # Test 5: Non-aligned size
    for n in [1, 7, 33, 127, 999, 2047, 3333, 99999]:
        x = torch.randn(n, device=device)
        out = bh_ops.scan_then_fan(x)
        ref = torch.cumsum(x, dim=0)
        ok = torch.allclose(out, ref, atol=1e-3)
        print(f"Test  n={n:6d}: max_err={torch.max(torch.abs(out - ref)).item():.2e}, PASS={ok}")

    # Test 6: Multi-dimensional (kernel flattens to 1D)
    x = torch.randn(32, 64, device=device)
    out = bh_ops.scan_then_fan(x)
    ref = torch.cumsum(x.reshape(-1), dim=0)
    print(f"Test6 (32x64):  max_err={torch.max(torch.abs(out - ref)).item():.2e}, PASS={torch.allclose(out, ref, atol=1e-4)}")

if __name__ == "__main__":
    test_prefix_sum()
