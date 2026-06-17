import torch
from bh_ops import navie_gemm, smem_gemm, smem_t8x8_gemm

def test():
    device = "cuda"

    test_cases = [
        # (M, K, N)
        (32, 32, 32),       # 整除 tile
        (64, 64, 64),       # 整除 tile
        (48, 32, 64),       # 不整除 tile
        (100, 50, 80),      # 不整除 tile，非2幂
        (1, 1, 1),          # 最小
        (256, 128, 256),    # 较大
    ]

    all_pass = True
    for M, K, N in test_cases:
        A = torch.randn(M, K, device=device)
        B = torch.randn(K, N, device=device)
        C_ref = torch.mm(A, B)

        # Naive GEMM
        C_naive = navie_gemm(A, B)
        ok_naive = torch.allclose(C_naive, C_ref, atol=1e-4)
        err_naive = (C_naive - C_ref).abs().max().item()

        # SMEM GEMM
        C_smem = smem_gemm(A, B)
        ok_smem = torch.allclose(C_smem, C_ref, atol=1e-4)
        err_smem = (C_smem - C_ref).abs().max().item()

        # SMEM Tile 8x8 GEMM
        C_t8x8 = smem_t8x8_gemm(A, B)
        ok_t8x8 = torch.allclose(C_t8x8, C_ref, atol=1e-4)
        err_t8x8 = (C_t8x8 - C_ref).abs().max().item()

        status_naive = "PASS" if ok_naive else "FAIL"
        status_smem = "PASS" if ok_smem else "FAIL"
        status_t8x8 = "PASS" if ok_t8x8 else "FAIL"
        if not ok_naive or not ok_smem or not ok_t8x8:
            all_pass = False

        print(f"  [{M}x{K}] x [{K}x{N}] = [{M}x{N}]: naive={status_naive}({err_naive:.2e})  smem={status_smem}({err_smem:.2e})  t8x8={status_t8x8}({err_t8x8:.2e})")

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")

def benchmark():
    device = "cuda"
    M, K, N = 512, 1024, 1024
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    num_iters = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    flops = 2.0 * M * K * N

    # Naive GEMM
    for _ in range(5):
        navie_gemm(A, B)
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        start.record()
        navie_gemm(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    naive_ms = sum(times) / len(times)
    naive_tflops = flops / (naive_ms / 1000.0) / 1e12

    # SMEM GEMM
    for _ in range(5):
        smem_gemm(A, B)
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        start.record()
        smem_gemm(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    smem_ms = sum(times) / len(times)
    smem_tflops = flops / (smem_ms / 1000.0) / 1e12

    # cuBLAS
    for _ in range(5):
        torch.mm(A, B)
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        start.record()
        torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    cublas_ms = sum(times) / len(times)
    cublas_tflops = flops / (cublas_ms / 1000.0) / 1e12

    print(f"\n{'='*70}")
    print(f"GEMM Benchmark [{M}x{K}] x [{K}x{N}]")
    print(f"{'='*70}")
    print(f"  Naive GEMM:  {naive_ms:.3f} ms  ({naive_tflops:.2f} TFLOPS)")
    print(f"  SMEM GEMM:   {smem_ms:.3f} ms  ({smem_tflops:.2f} TFLOPS)")
    print(f"  cuBLAS:      {cublas_ms:.3f} ms  ({cublas_tflops:.2f} TFLOPS)")
    print(f"  SMEM/Naive:  {naive_ms/smem_ms:.2f}x")
    print(f"  SMEM/cuBLAS: {cublas_ms/smem_ms:.2f}x")

def ncu_benchmark():
    """单次运行，用于 ncu profiling"""
    device = "cuda"
    M, K, N = 512, 1024, 1024
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C = smem_gemm(A, B)
    # C = navie_gemm(A,B)
    torch.cuda.synchronize()

if __name__ == "__main__":
    test()
