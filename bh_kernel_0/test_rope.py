import torch
from bh_ops import rope_decode, rope_qkv_decode, rope_qkv_decode_smem


# ============================================================
# cos/sin 表构造
# ============================================================
def build_rope_table(max_seq_len: int, head_dim: int, device="cuda"):
    """
    构造 RoPE 的 cos/sin 查找表，所有序列共享，只需构造一次。

    原理:
        theta_i = 10000^(-2i/d),  i = 0,1,...,half_dim-1
        每个位置 m 的旋转角度 = m * theta_i
        cos_table[m, i] = cos(m * theta_i)
        sin_table[m, i] = sin(m * theta_i)
    """
    half_dim = head_dim // 2

    # theta_i = 10000^(-2i/d), 值从 1.0 递减到 ~0.0001
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    # 位置序列
    positions = torch.arange(max_seq_len, dtype=torch.float32)

    # 外积: freqs[m, i] = m * theta_i，即位置 m 在第 i 个 pair 的旋转角度
    freqs = torch.outer(positions, theta)

    cos_table = torch.cos(freqs)  # [max_seq_len, half_dim]
    sin_table = torch.sin(freqs)  # [max_seq_len, half_dim]

    return cos_table.to(device), sin_table.to(device)


# ============================================================
# PyTorch 参考实现
# ============================================================
def apply_rope_decode_ref(x, positions, cos_table, sin_table):
    """
    前后半配对 RoPE 参考实现。
    x: [batch, num_heads, head_dim]
    """
    batch, num_heads, head_dim = x.shape
    half_dim = head_dim // 2

    pos = positions.unsqueeze(1).expand(-1, half_dim)
    cos = cos_table.gather(0, pos).unsqueeze(1).expand(-1, num_heads, -1)
    sin = sin_table.gather(0, pos).unsqueeze(1).expand(-1, num_heads, -1)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    out = torch.empty_like(x)
    out[..., :half_dim]  = x1 * cos - x2 * sin
    out[..., half_dim:]  = x1 * sin + x2 * cos
    return out


def apply_rope_qkv_ref(x, positions, cos_table, sin_table, num_kv_heads):
    """
    相邻配对 QKV RoPE 参考实现。
    x: [batch, num_heads + 2*num_kv_heads, head_dim]
    Q: [:, :num_heads, :]
    K: [:, num_heads:num_heads+num_kv_heads, :]
    V: [:, num_heads+num_kv_heads:, :]  (不处理)
    """
    batch = x.size(0)
    num_heads = x.size(1) - 2 * num_kv_heads
    head_dim = x.size(2)
    half_dim = head_dim // 2

    out = x.clone()

    pos = positions.unsqueeze(1).expand(-1, half_dim)  # [batch, half_dim]

    # 对 Q 和 K 做相邻配对 RoPE
    for qk_idx in range(num_heads + num_kv_heads):
        # 查 cos/sin
        cos = cos_table.gather(0, pos)  # [batch, half_dim]
        sin = sin_table.gather(0, pos)  # [batch, half_dim]
        cos = cos.unsqueeze(1)  # [batch, 1, half_dim]
        sin = sin.unsqueeze(1)

        # 取这个 head 的数据: [batch, 1, head_dim]
        head_data = out[:, qk_idx:qk_idx+1, :]

        # 相邻配对: (x[2i], x[2i+1]) 用 cos[i], sin[i]
        x_even = head_data[:, :, 0::2]  # [batch, 1, half_dim]
        x_odd  = head_data[:, :, 1::2]  # [batch, 1, half_dim]

        rotated = torch.empty_like(head_data)
        rotated[:, :, 0::2] = x_even * cos - x_odd * sin
        rotated[:, :, 1::2] = x_even * sin + x_odd * cos
        out[:, qk_idx:qk_idx+1, :] = rotated

    # V 不处理，已经在 clone 中保留
    return out


# ============================================================
# 正确性测试
# ============================================================
def test_correctness():
    device = "cuda"
    max_seq_len = 512

    # --- 前后半配对 rope_decode ---
    print("=" * 60)
    print("rope_decode (前后半配对) Correctness Tests")
    print("=" * 60)

    test_cases = [
        # (1, 8, 64),
        (4, 16, 128),
        # (8, 32, 64),
        # (16, 8, 256),
    ]

    all_pass = True
    for batch, num_heads, head_dim in test_cases:
        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
        x = torch.randn(batch, num_heads, head_dim, device=device)
        positions = torch.randint(0, max_seq_len, (batch,), device=device, dtype=torch.int32)

        x_copy = x.clone()
        out_cuda = rope_decode(x_copy, positions, cos_table, sin_table)
        out_ref = apply_rope_decode_ref(x, positions, cos_table, sin_table)

        ok = torch.allclose(out_cuda, out_ref, atol=1e-5)
        if not ok:
            max_err = (out_cuda - out_ref).abs().max().item()
            print(f"  [batch={batch}, heads={num_heads}, dim={head_dim}]: FAIL  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [batch={batch}, heads={num_heads}, dim={head_dim}]: PASS")

    if all_pass:
        print("All rope_decode tests passed!")
    else:
        print("Some rope_decode tests FAILED!")

    # --- 相邻配对 rope_qkv_decode (float4) ---
    print()
    print("=" * 60)
    print("rope_qkv_decode (相邻配对 + float4 + GQA) Correctness Tests")
    print("=" * 60)

    qkv_test_cases = [
        # (batch, num_heads, num_kv_heads, head_dim)
        (2, 8, 8, 64),       # MHA: num_kv_heads == num_heads
        (4, 16, 4, 128),     # GQA: 4x ratio
        (8, 32, 8, 128),     # GQA: 4x ratio (LLaMA-2 style)
        (4, 32, 32, 64),     # MHA
    ]

    all_pass = True
    for batch, num_heads, num_kv_heads, head_dim in qkv_test_cases:
        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
        total_qkv_heads = num_heads + 2 * num_kv_heads
        x = torch.randn(batch, total_qkv_heads, head_dim, device=device)
        positions = torch.randint(0, max_seq_len, (batch,), device=device, dtype=torch.int32)

        # float4 版本
        x_copy = x.clone()
        out_cuda = rope_qkv_decode(x_copy, positions, cos_table, sin_table, num_kv_heads)
        out_ref = apply_rope_qkv_ref(x, positions, cos_table, sin_table, num_kv_heads)

        ok = torch.allclose(out_cuda, out_ref, atol=1e-5)
        if not ok:
            max_err = (out_cuda - out_ref).abs().max().item()
            print(f"  [batch={batch}, h={num_heads}, kv={num_kv_heads}, dim={head_dim}]: FAIL  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [batch={batch}, h={num_heads}, kv={num_kv_heads}, dim={head_dim}]: PASS")

        # 验证 V 没有被修改
        v_start = num_heads + num_kv_heads
        v_end = num_heads + 2 * num_kv_heads
        v_ok = torch.allclose(out_cuda[:, v_start:v_end, :], x[:, v_start:v_end, :], atol=1e-7)
        if not v_ok:
            print(f"    V was modified! This is a bug.")

    if all_pass:
        print("All rope_qkv_decode tests passed!")
    else:
        print("Some rope_qkv_decode tests FAILED!")

    # --- 相邻配对 rope_qkv_decode_smem ---
    print()
    print("=" * 60)
    print("rope_qkv_decode_smem (相邻配对 + smem + GQA) Correctness Tests")
    print("=" * 60)

    all_pass = True
    for batch, num_heads, num_kv_heads, head_dim in qkv_test_cases:
        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
        total_qkv_heads = num_heads + 2 * num_kv_heads
        x = torch.randn(batch, total_qkv_heads, head_dim, device=device)
        positions = torch.randint(0, max_seq_len, (batch,), device=device, dtype=torch.int32)

        x_copy = x.clone()
        out_cuda = rope_qkv_decode_smem(x_copy, positions, cos_table, sin_table, num_kv_heads)
        out_ref = apply_rope_qkv_ref(x, positions, cos_table, sin_table, num_kv_heads)

        ok = torch.allclose(out_cuda, out_ref, atol=1e-5)
        if not ok:
            max_err = (out_cuda - out_ref).abs().max().item()
            print(f"  [batch={batch}, h={num_heads}, kv={num_kv_heads}, dim={head_dim}]: FAIL  max_err={max_err:.2e}")
            all_pass = False
        else:
            print(f"  [batch={batch}, h={num_heads}, kv={num_kv_heads}, dim={head_dim}]: PASS")

    if all_pass:
        print("All rope_qkv_decode_smem tests passed!")
    else:
        print("Some rope_qkv_decode_smem tests FAILED!")


# ============================================================
# 性能测试
# ============================================================
def test_performance():
    device = "cuda"
    max_seq_len = 8192

    print()
    print("=" * 60)
    print("RoPE Decode Performance (20 runs, avg)")
    print("=" * 60)

    # --- rope_decode (前后半配对) ---
    print()
    print("--- rope_decode (前后半配对) ---")
    print(f"  {'shape':>24s}  {'kernel(ms)':>10s}  {'torch(ms)':>10s}  {'speedup':>8s}")
    print("  " + "-" * 58)

    for batch, num_heads, head_dim in [(32, 32, 128), (128, 32, 128), (256, 16, 64)]:
        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
        x = torch.randn(batch, num_heads, head_dim, device=device)
        positions = torch.randint(0, 1000, (batch,), device=device, dtype=torch.int32)

        for _ in range(5):
            rope_decode(x.clone(), positions, cos_table, sin_table)
            apply_rope_decode_ref(x, positions, cos_table, sin_table)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_kernel = []
        for _ in range(20):
            x_copy = x.clone()
            start.record()
            out_k = rope_decode(x_copy, positions, cos_table, sin_table)
            end.record()
            torch.cuda.synchronize()
            times_kernel.append(start.elapsed_time(end))

        times_torch = []
        for _ in range(20):
            start.record()
            out_t = apply_rope_decode_ref(x, positions, cos_table, sin_table)
            end.record()
            torch.cuda.synchronize()
            times_torch.append(start.elapsed_time(end))

        kernel_ms = sum(times_kernel) / len(times_kernel)
        torch_ms = sum(times_torch) / len(times_torch)
        speedup = torch_ms / kernel_ms

        ok = torch.allclose(out_k, out_t, atol=1e-4)
        shape_str = f"[{batch},{num_heads},{head_dim}]"
        print(f"  {shape_str:>24s}  {kernel_ms:>10.3f}  {torch_ms:>10.3f}  {speedup:>7.2f}x  {'OK' if ok else 'FAIL'}")

    # --- rope_qkv_decode (float4 vs smem) ---
    print()
    print("--- rope_qkv_decode: float4 vs smem ---")
    print(f"  {'config':>30s}  {'float4(ms)':>10s}  {'smem(ms)':>10s}  {'ratio':>8s}")
    print("  " + "-" * 64)

    for batch, num_heads, num_kv_heads, head_dim in [
        (32, 32, 8, 128),
        (128, 32, 8, 128),
        (64, 64, 8, 128),
    ]:
        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
        total_qkv_heads = num_heads + 2 * num_kv_heads
        x = torch.randn(batch, total_qkv_heads, head_dim, device=device)
        positions = torch.randint(0, 1000, (batch,), device=device, dtype=torch.int32)

        # warmup
        for _ in range(5):
            rope_qkv_decode(x.clone(), positions, cos_table, sin_table, num_kv_heads)
            rope_qkv_decode_smem(x.clone(), positions, cos_table, sin_table, num_kv_heads)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # float4
        times_f4 = []
        for _ in range(20):
            x_copy = x.clone()
            start.record()
            out_f4 = rope_qkv_decode(x_copy, positions, cos_table, sin_table, num_kv_heads)
            end.record()
            torch.cuda.synchronize()
            times_f4.append(start.elapsed_time(end))

        # smem
        times_smem = []
        for _ in range(20):
            x_copy = x.clone()
            start.record()
            out_smem = rope_qkv_decode_smem(x_copy, positions, cos_table, sin_table, num_kv_heads)
            end.record()
            torch.cuda.synchronize()
            times_smem.append(start.elapsed_time(end))

        f4_ms = sum(times_f4) / len(times_f4)
        smem_ms = sum(times_smem) / len(times_smem)
        ratio = smem_ms / f4_ms

        ok = torch.allclose(out_f4, out_smem, atol=1e-4)
        config_str = f"[b={batch},h={num_heads},kv={num_kv_heads},d={head_dim}]"
        print(f"  {config_str:>30s}  {f4_ms:>10.3f}  {smem_ms:>10.3f}  {ratio:>7.2f}x  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    test_correctness()
    test_performance()
