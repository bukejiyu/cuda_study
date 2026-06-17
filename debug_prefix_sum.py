import torch
import bh_ops

device = "cuda"

# Simple debug: small input, compare element by element
torch.manual_seed(42)

# Test with n=33 (fails in earlier test)
x = torch.randn(33, device=device)
out = bh_ops.scan_then_fan(x)
ref = torch.cumsum(x, dim=0)

print("Input:", x[:10].cpu().numpy())
print("Expected:", ref[:10].cpu().numpy())
print("Got:     ", out[:10].cpu().numpy())
print("Diff:    ", (out - ref)[:10].cpu().numpy())
print()

# Test n=128 (exactly kThreadNum elements)
x = torch.randn(128, device=device)
out = bh_ops.scan_then_fan(x)
ref = torch.cumsum(x, dim=0)
print(f"n=128: max_err={torch.max(torch.abs(out-ref)).item():.2e}")

# Test n=256 (one iteration of inner loop per block)
x = torch.randn(256, device=device)
out = bh_ops.scan_then_fan(x)
ref = torch.cumsum(x, dim=0)
print(f"n=256: max_err={torch.max(torch.abs(out-ref)).item():.2e}")

# Test n=1024 (exactly perBlockNum)
x = torch.randn(1024, device=device)
out = bh_ops.scan_then_fan(x)
ref = torch.cumsum(x, dim=0)
print(f"n=1024: max_err={torch.max(torch.abs(out-ref)).item():.2e}")

# Test n=1025 (just over one block)
x = torch.randn(1025, device=device)
out = bh_ops.scan_then_fan(x)
ref = torch.cumsum(x, dim=0)
print(f"n=1025: max_err={torch.max(torch.abs(out-ref)).item():.2e}")
# Show where the error is
diff = torch.abs(out - ref)
print(f"  Error starts at idx={diff.argmax().item()}, err={diff.max().item():.4f}")
print(f"  out[1023]={out[1023].item():.4f}, ref[1023]={ref[1023].item():.4f}")
print(f"  out[1024]={out[1024].item():.4f}, ref[1024]={ref[1024].item():.4f}")
