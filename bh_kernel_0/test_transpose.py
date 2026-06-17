import torch
from bh_ops import transpose_v3 as transpose


def test_correctness():
    device = "cuda"
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    # 不同 [N, D] 组合，覆盖 TILE_DIM=32 边界情况
    test_cases = [
        (1, 1),       # 最小输入
        (1, 7),       # 非 TILE_DIM 倍数
        (1, 32),      # 恰好等于 TILE_DIM
        (1, 100),     # 略大于 TILE_DIM
        (4, 7),       # 多行，非 TILE_DIM 倍数
        (4, 32),      # 多行，N/D = TILE_DIM
        (4, 64),      # 多行，2*TILE_DIM
        (8, 100),     # 多行，非对齐
        (16, 1024),   # 常见规模
        (33, 65),     # 非对齐，测试边界 padding
        (128, 256),   # 中等规模
    ]

    all_pass = True
    for N, D in test_cases:
        x = torch.randn(N, D, device=device)
        out = transpose(x)
        ref = x.t()
        ok = torch.allclose(out, ref, atol=1e-6)
        if not ok:
            max_err = (out - ref).abs().max().item()
            print(f"  [N={N}, D={D}]: FAIL  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [N={N}, D={D}]: PASS")

    # 边界：全零
    print()
    print("--- Edge case: all zeros ---")
    x = torch.zeros(4, 32, device=device)
    out = transpose(x)
    ref = x.t()
    ok = torch.allclose(out, ref, atol=1e-6)
    print(f"  all zeros: {'PASS' if ok else 'FAIL'}")

    # 边界：全1
    print()
    print("--- Edge case: all ones ---")
    x = torch.ones(8, 64, device=device)
    out = transpose(x)
    ref = x.t()
    ok = torch.allclose(out, ref, atol=1e-6)
    print(f"  all ones: {'PASS' if ok else 'FAIL'}")

    # 边界：含大数 / 极小数（transpose 是精确操作，应无损）
    print()
    print("--- Numerical precision (large/small values) ---")
    x = torch.randn(4, 64, device=device) * 1e6
    out = transpose(x)
    ref = x.t()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-6)
    print(f"  large values: {'PASS' if ok else 'FAIL'}  max_err={(out - ref).abs().max().item():.2e}")

    x = torch.randn(4, 64, device=device) * 1e-6
    out = transpose(x)
    ref = x.t()
    ok = torch.allclose(out, ref, atol=1e-12)
    print(f"  small values: {'PASS' if ok else 'FAIL'}  max_err={(out - ref).abs().max().item():.2e}")

    # 输出形状检查
    print()
    print("--- Shape check ---")
    x = torch.randn(7, 13, device=device)
    out = transpose(x)
    shape_ok = (out.shape == (13, 7))
    print(f"  input {x.shape} -> output {out.shape}: {'PASS' if shape_ok else 'FAIL'}")

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
    print(f"  {'[N,D]':>20s}  {'kernel(ms)':>10s}  {'torch(ms)':>10s}  {'speedup':>8s}  {'bandwidth(GB/s)':>16s}  {'OK':>5s}")
    print("  " + "-" * 80)

    for N, D in [(256, 256), (1024, 1024), (4096, 4096), (8192, 8192), (16384, 16384)]:
        x = torch.randn(N, D, device=device, dtype=torch.float32)

        # warmup
        for _ in range(5):
            transpose(x)
            x.t().contiguous()
        torch.cuda.synchronize()

        # kernel timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_kernel = []
        for _ in range(20):
            start.record()
            out_k = transpose(x)
            end.record()
            torch.cuda.synchronize()
            times_kernel.append(start.elapsed_time(end))

        # torch timing
        times_torch = []
        for _ in range(20):
            start.record()
            out_t = x.t().contiguous()
            end.record()
            torch.cuda.synchronize()
            times_torch.append(start.elapsed_time(end))

        kernel_ms = sum(times_kernel) / len(times_kernel)
        torch_ms = sum(times_torch) / len(times_torch)
        speedup = torch_ms / kernel_ms

        # 带宽: 读 N*D 个 float + 写 D*N 个 float = 2 * N * D * 4 bytes
        bytes_moved = 2 * N * D * 4
        bandwidth_gbs = bytes_moved / (kernel_ms * 1e-3) / 1e9

        ok = torch.allclose(out_k, out_t, atol=1e-5)
        label = f"[{N},{D}]"
        print(f"  {label:>20s}  {kernel_ms:>10.3f}  {torch_ms:>10.3f}  {speedup:>7.2f}x  {bandwidth_gbs:>15.2f}  {'OK' if ok else 'FAIL':>5s}")


def test_ncu():
    """专门跑 ncu 的隔离测试，只调一次 kernel，无多余操作"""
    device = "cuda"
    N, D = 8192, 8192
    x = torch.randn(N, D, device=device, dtype=torch.float32)
    # warmup
    transpose(x)
    torch.cuda.synchronize()
    # 只跑一次，方便 ncu 采集
    out = transpose(x)
    torch.cuda.synchronize()
    print(f"ncu test done: input [{N},{D}] -> output {list(out.shape)}")


if __name__ == "__main__":
    # test_correctness()
    # test_performance()
    test_ncu()
