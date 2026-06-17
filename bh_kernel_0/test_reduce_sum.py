import torch
from bh_ops import reduce_sum


def timed_run(func, *args, **kwargs):
    """用 CUDA Event 计时，返回 (结果, 耗时ms)"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # # warmup
    # for _ in range(5):
    #     func(*args, **kwargs)
    # torch.cuda.synchronize()
    # 多次测量取平均
    times = []
    for _ in range(1):
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    avg_ms = sum(times) / len(times)
    return result, avg_ms


def test_correctness():
    device = "cuda"
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    # 不同 [N, D] 组合，覆盖边界情况
    test_cases = [
        (1, 1),       # 最小输入
        (1, 7),       # D 不是 kPerBlock 的倍数
        (1, 128),     # D 较小
        (1, 1024),    # D = kPerBlock
        (1, 1025),    # D > kPerBlock，需要多个 block.y
        (4, 7),       # 多行，D 不是 kPerBlock 的倍数
        (4, 1024),    # 多行，D = kPerBlock
        (8, 2000),    # 多行，D 较大
        (16, 4096),   # 常见规模
        (128, 768),   # 类似 hidden_size
        # (8192,8192)
    ]

    all_pass = True
    for N, D in test_cases:
        x = torch.randn(N, D, device=device)
        out = reduce_sum(x)
        ref = x.sum(dim=1)
        ok = torch.allclose(out, ref, atol=1e-3)
        if not ok:
            max_err = (out - ref).abs().max().item()
            print(f"  [N={N}, D={D}]: FAIL  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [N={N}, D={D}]: PASS")

    # 数值稳定性：全零
    print()
    print("--- Edge case: all zeros ---")
    x = torch.zeros(4, 1024, device=device)
    out = reduce_sum(x)
    ref = x.sum(dim=1)
    ok = torch.allclose(out, ref, atol=1e-6)
    print(f"  all zeros: {'PASS' if ok else 'FAIL'}")

    # 数值稳定性：含大数
    print()
    print("--- Numerical stability (large values) ---")
    x = torch.randn(4, 1024, device=device) * 1e4
    out = reduce_sum(x)
    ref = x.sum(dim=1)
    ok = torch.allclose(out, ref, atol=1e0, rtol=1e-5)
    max_err = (out - ref).abs().max().item()
    print(f"  large values: {'PASS' if ok else 'FAIL'}  max_err={max_err:.2e}")

    # 数值稳定性：全为1
    print()
    print("--- Edge case: all ones ---")
    N, D = 4, 5000
    x = torch.ones(N, D, device=device)
    out = reduce_sum(x)
    ref = x.sum(dim=1)
    ok = torch.allclose(out, ref, atol=1e-3)
    print(f"  all ones [N={N}, D={D}]: {'PASS' if ok else 'FAIL'}  (expect {D}, got {out[0].item():.2f})")

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
    print(f"  {'[N,D]':>20s}  {'kernel(ms)':>10s}  {'torch(ms)':>10s}  {'speedup':>8s}  {'bandwidth(GB/s)':>16s}  {'PASS':>5s}")
    print("  " + "-" * 80)

    # for N, D in [(1024, 256), (4096, 512), (8192, 1024), (16384, 2048), (32768, 4096)]:
    for N, D in [(8192, 8192)]:

        x = torch.randn(N, D, device=device, dtype=torch.float32)

        # 自定义 kernel
        out_kernel, kernel_ms = timed_run(reduce_sum, x)
        print(kernel_ms)
        # # torch.sum 作为基准
        # out_torch, torch_ms = timed_run(lambda t: t.sum(dim=1), x)

        # # 正确性
        # ok = torch.allclose(out_kernel, out_torch, atol=1e-2)

        # # 带宽计算: 读 N*D 个 float + 写 N 个 float = (N*D + N) * 4 bytes
        # bytes_moved = (N * D + N) * 4
        # bandwidth_gbs = bytes_moved / (kernel_ms * 1e-3) / 1e9

        # speedup = torch_ms / kernel_ms
        # label = f"[{N},{D}]"
        # print(f"  {label:>20s}  {kernel_ms:>10.3f}  {torch_ms:>10.3f}  {speedup:>7.2f}x  {bandwidth_gbs:>15.2f}  {'OK' if ok else 'FAIL':>5s}")


if __name__ == "__main__":
    # test_correctness()
    test_performance()
