import torch
import torch.nn as nn
from bh_ops import group_norm_v2 as group_norm


def test_correctness():
    device = "cuda"
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    # 不同 [N, C, H, W, num_groups] 组合
    test_cases = [
        (1, 4, 8, 8, 2),      # 小输入
        (2, 4, 16, 16, 2),    # 常见配置
        (1, 8, 32, 32, 4),    # 更多通道
        (2, 16, 64, 64, 8),   # 典型 CNN 规模
        (4, 32, 16, 16, 16),  # 大 batch
        (1, 6, 8, 8, 3),      # C=6, G=2
        (1, 4, 8, 8, 1),      # num_groups=1 → Layer Norm
        (1, 4, 8, 8, 4),      # num_groups=C → Instance Norm
    ]

    all_pass = True
    for N, C, H, W, ng in test_cases:
        x = torch.randn(N, C, H, W, device=device)

        # 不带 affine
        out = group_norm(x, ng, None, None)
        ref = nn.functional.group_norm(x, ng)
        ok = torch.allclose(out, ref, atol=1e-4)
        status = "PASS" if ok else "FAIL"
        if not ok:
            max_err = (out - ref).abs().max().item()
            print(f"  [N={N},C={C},H={H},W={W},G={ng}] no affine: {status}  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [N={N},C={C},H={H},W={W},G={ng}] no affine: {status}")

        # 带 affine
        gamma = torch.randn(C, device=device)
        beta = torch.randn(C, device=device)
        out = group_norm(x, ng, gamma, beta)
        ref = nn.functional.group_norm(x, ng, gamma, beta)
        ok = torch.allclose(out, ref, atol=1e-4)
        status = "PASS" if ok else "FAIL"
        if not ok:
            max_err = (out - ref).abs().max().item()
            print(f"  [N={N},C={C},H={H},W={W},G={ng}] w/ affine: {status}  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [N={N},C={C},H={H},W={W},G={ng}] w/ affine: {status}")

    if all_pass:
        print("\nAll correctness tests passed!")
    else:
        print("\nSome tests FAILED!")


def test_performance():
    device = "cuda"
    print()
    print("=" * 60)
    print("Performance Tests (20 runs, avg)")
    print("=" * 60)
    print(f"  {'[N,C,H,W,G]':>28s}  {'kernel(ms)':>10s}  {'torch(ms)':>10s}  {'speedup':>8s}")
    print("  " + "-" * 62)

    for N, C, H, W, ng in [(2, 32, 64, 64, 8), (4, 64, 32, 32, 16), (8, 32, 128, 128, 8)]:
        x = torch.randn(N, C, H, W, device=device)
        gamma = torch.randn(C, device=device)
        beta = torch.randn(C, device=device)

        # warmup
        for _ in range(5):
            group_norm(x, ng, gamma, beta)
            nn.functional.group_norm(x, ng, gamma, beta)
        torch.cuda.synchronize()

        # kernel
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_kernel = []
        for _ in range(20):
            start.record()
            out_k = group_norm(x, ng, gamma, beta)
            end.record()
            torch.cuda.synchronize()
            times_kernel.append(start.elapsed_time(end))

        # torch
        times_torch = []
        for _ in range(20):
            start.record()
            out_t = nn.functional.group_norm(x, ng, gamma, beta)
            end.record()
            torch.cuda.synchronize()
            times_torch.append(start.elapsed_time(end))

        kernel_ms = sum(times_kernel) / len(times_kernel)
        torch_ms = sum(times_torch) / len(times_torch)
        speedup = torch_ms / kernel_ms

        ok = torch.allclose(out_k, out_t, atol=1e-4)
        label = f"[{N},{C},{H},{W},{ng}]"
        print(f"  {label:>28s}  {kernel_ms:>10.3f}  {torch_ms:>10.3f}  {speedup:>7.2f}x  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    test_correctness()
    test_performance()
