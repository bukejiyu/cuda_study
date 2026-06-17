import torch
from bh_ops import softmax


def test_correctness():
    device = "cuda"
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    # 不同 [N, D] 组合，覆盖边界情况
    test_cases = [
        (1, 1),       # 最小输入
        (1, 7),       # D 不是 kThreads 的倍数
        (1, 128),     # D 恰好等于 kThreads
        (1, 200),     # D > kThreads，需要多轮迭代
        (4, 7),       # 多行，D 不是 kThreads 的倍数
        (4, 128),     # 多行，D = kThreads
        (4, 256),     # 多行，多轮迭代
        (8, 500),     # 多行，D 较大
        (16, 1024),   # 常见规模
    ]

    all_pass = True
    for N, D in test_cases:
        x = torch.randn(N, D, device=device)
        out = softmax(x)
        ref = torch.softmax(x, dim=1)
        ok = torch.allclose(out, ref, atol=1e-5)
        if not ok:
            max_err = (out - ref).abs().max().item()
            print(f"  [N={N}, D={D}]: FAIL  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [N={N}, D={D}]: PASS")

    # 数值稳定性：含大数
    print()
    print("--- Numerical stability (large values) ---")
    x = torch.tensor([[100.0, 200.0, 300.0, 400.0]], device=device)
    out = softmax(x)
    ref = torch.softmax(x, dim=1)
    ok = torch.allclose(out, ref, atol=1e-5)
    print(f"  large values: {'PASS' if ok else 'FAIL'}")

    # 数值稳定性：含负大数
    x = torch.tensor([[-100.0, -200.0, -300.0, -400.0]], device=device)
    out = softmax(x)
    ref = torch.softmax(x, dim=1)
    ok = torch.allclose(out, ref, atol=1e-5)
    print(f"  negative large: {'PASS' if ok else 'FAIL'}")

    # 概率和应为 1
    print()
    print("--- Sum-to-1 check ---")
    x = torch.randn(8, 1024, device=device)
    out = softmax(x)
    row_sums = out.sum(dim=1)
    ok = torch.allclose(row_sums, torch.ones(8, device=device), atol=1e-5)
    print(f"  row sums == 1: {'PASS' if ok else 'FAIL'}  max_err={((row_sums - 1).abs().max().item()):.2e}")

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
    print(f"  {'[N,D]':>16s}  {'kernel(ms)':>10s}  {'torch(ms)':>10s}  {'speedup':>8s}")
    print("  " + "-" * 50)

    for N, D in [(1024, 256), (4096, 512), (8192, 1024), (16384, 2048)]:
        x = torch.randn(N, D, device=device)

        # warmup
        for _ in range(5):
            softmax(x)
            torch.softmax(x, dim=1)
        torch.cuda.synchronize()

        # kernel
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_kernel = []
        for _ in range(20):
            start.record()
            out_k = softmax(x)
            end.record()
            torch.cuda.synchronize()
            times_kernel.append(start.elapsed_time(end))

        # torch
        times_torch = []
        for _ in range(20):
            start.record()
            out_t = torch.softmax(x, dim=1)
            end.record()
            torch.cuda.synchronize()
            times_torch.append(start.elapsed_time(end))

        kernel_ms = sum(times_kernel) / len(times_kernel)
        torch_ms = sum(times_torch) / len(times_torch)
        speedup = torch_ms / kernel_ms

        ok = torch.allclose(out_k, out_t, atol=1e-4)
        print(f"  [{N},{D}]{'':>{16-len(str(N))-len(str(D))-3}s}  {kernel_ms:>10.3f}  {torch_ms:>10.3f}  {speedup:>7.2f}x  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    test_correctness()
    test_performance()
