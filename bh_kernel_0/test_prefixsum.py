import sys
import torch
from bh_ops import scan_then_fan
# sys.path.insert(0, '../bh_ops_py')
# from _C import scan_then_fan


def timed_run(func, *args, **kwargs):
    """用 CUDA Event 计时，返回 (结果, 耗时ms)"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    # 多次测量取平均
    times = []
    for _ in range(20):
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    avg_ms = sum(times) / len(times)
    return result, avg_ms


def test_correctness_and_perf():
    device = "cuda"

    # ============ 小输入正确性 ============
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    for n in [1, 7, 33, 127, 999, 1024, 5000, 99999]:
        x = torch.randn(n, device=device)
        out = scan_then_fan(x)
        ref = torch.cumsum(x, dim=0)
        ok = torch.allclose(out, ref, atol=1e-3)
        print(f"  n={n:>8}: PASS={ok}")

    # ============ 大输入正确性 + 稳定性 ============
    print()
    print("=" * 60)
    print("Large Input Stability Tests (50 trials each)")
    print("=" * 60)

    for n in [1_000_000, 4_000_000, 10_000_000, 50_000_000]:
        fails = 0
        max_err = 0
        for trial in range(50):
            x = torch.randn(n, device=device)
            out = scan_then_fan(x)
            ref = torch.cumsum(x, dim=0)
            err = torch.max(torch.abs(out - ref)).item()
            max_err = max(max_err, err)
            if err > 1.0:
                fails += 1
        print(f"  n={n:>12,}: {fails}/50 failures, max_err={max_err:.2e}")

    # ============ 性能测试 ============
    print()
    print("=" * 60)
    print("Performance Tests (20 runs, avg)")
    print("=" * 60)
    print(f"  {'n':>12s}  {'kernel(ms)':>10s}  {'torch(ms)':>10s}  {'speedup':>8s}  {'bandwidth(GB/s)':>16s}  {'PASS':>5s}")
    print("  " + "-" * 70)

    for n in [100_000, 1_000_000, 4_000_000, 10_000_000, 50_000_000, 100_000_000]:
        x = torch.randn(n, device=device, dtype=torch.float32)

        # 自定义 kernel
        out_kernel, kernel_ms = timed_run(scan_then_fan, x)

        # torch.cumsum 作为基准
        out_torch, torch_ms = timed_run(torch.cumsum, x, dim=0)

        # 正确性
        ok = torch.allclose(out_kernel, out_torch, atol=1e-2)

        # 带宽计算: 前缀和读 n 个 float + 写 n 个 float = 2 * n * 4 bytes
        bytes_moved = 2 * n * 4
        bandwidth_gbs = bytes_moved / (kernel_ms * 1e-3) / 1e9

        speedup = torch_ms / kernel_ms
        print(f"  {n:>12,}  {kernel_ms:>10.3f}  {torch_ms:>10.3f}  {speedup:>7.2f}x  {bandwidth_gbs:>15.2f}  {'OK' if ok else 'FAIL':>5s}")


if __name__ == "__main__":
    test_correctness_and_perf()
