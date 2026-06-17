import torch
from bh_ops import decode_apply_rope, rope_qkv_decode, rope_qkv_decode_smem

def build_rope_table(max_seq_len, head_dim, device="cuda"):
    half_dim = head_dim // 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    cos_table = torch.cos(freqs)
    sin_table = torch.sin(freqs)
    return cos_table.to(device), sin_table.to(device)

def apply_rope_qkv_ref(x, positions, cos_table, sin_table, num_kv_heads):
    """
    相邻配对 QKV RoPE 参考实现。
    x: [token_len, num_heads + 2*num_kv_heads, head_dim]
    """
    token_len = x.size(0)
    num_heads = x.size(1) - 2 * num_kv_heads
    head_dim = x.size(2)
    half_dim = head_dim // 2

    out = x.clone()
    pos = positions.unsqueeze(1).expand(-1, half_dim)  # [token_len, half_dim]

    for qk_idx in range(num_heads + num_kv_heads):
        cos = cos_table.gather(0, pos)  # [token_len, half_dim]
        sin = sin_table.gather(0, pos)
        cos = cos.unsqueeze(1)  # [token_len, 1, half_dim]
        sin = sin.unsqueeze(1)

        head_data = out[:, qk_idx:qk_idx+1, :]
        x_even = head_data[:, :, 0::2]
        x_odd  = head_data[:, :, 1::2]

        rotated = torch.empty_like(head_data)
        rotated[:, :, 0::2] = x_even * cos - x_odd * sin
        rotated[:, :, 1::2] = x_even * sin + x_odd * cos
        out[:, qk_idx:qk_idx+1, :] = rotated

    return out

def test():
    max_seq_len = 512
    device = "cuda"

    test_cases = [
        # (token_len, num_heads, num_kv_heads, head_dim)
        (2, 8, 8, 128),       # MHA
        (4, 16, 4, 128),      # GQA
        (8, 32, 8, 128),      # GQA
        (1, 32, 32, 128),     # decode: token_len=1
    ]

    all_pass = True
    for token_len, num_heads, num_kv_heads, head_dim in test_cases:
        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
        total_qkv_heads = num_heads + 2 * num_kv_heads
        x = torch.randn(token_len, total_qkv_heads, head_dim, device=device)
        positions = torch.randint(0, max_seq_len, (token_len,), device=device, dtype=torch.int32)

        # CUDA kernel
        x_copy = x.clone()
        out_cuda = decode_apply_rope(x_copy, cos_table, sin_table, positions, num_heads, num_kv_heads, head_dim)

        # 参考实现
        out_ref = apply_rope_qkv_ref(x, positions, cos_table, sin_table, num_kv_heads)

        # 检查 Q 和 K 的精度
        ok = torch.allclose(out_cuda, out_ref, atol=1e-5)
        max_err = (out_cuda - out_ref).abs().max().item()

        # 检查 V 没有被修改
        v_start = num_heads + num_kv_heads
        v_end = num_heads + 2 * num_kv_heads
        v_ok = torch.allclose(out_cuda[:, v_start:v_end, :], x[:, v_start:v_end, :], atol=1e-7)

        status = "PASS" if (ok and v_ok) else "FAIL"
        if not ok or not v_ok:
            all_pass = False
        v_status = "" if v_ok else " [V was modified!]"

        print(f"  [t={token_len}, h={num_heads}, kv={num_kv_heads}, d={head_dim}]: {status}  max_err={max_err:.2e}{v_status}")

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")

def test_performance():
    max_seq_len = 8192
    device = "cuda"
    num_iters = 20

    print()
    print("=" * 80)
    print("RoPE Decode Performance (20 runs, avg)")
    print("=" * 80)
    print(f"  {'config':>30s}  {'rope0(ms)':>10s}  {'qkv_f4(ms)':>10s}  {'qkv_smem(ms)':>12s}  {'torch(ms)':>10s}")
    print("  " + "-" * 78)

    test_cases = [
        # (token_len, num_heads, num_kv_heads, head_dim)
        (1, 32, 8, 128),       # decode
        (1, 32, 32, 128),      # decode MHA
        (4, 32, 8, 128),       # prefill
        (8, 32, 8, 128),       # prefill
        (32, 32, 8, 128),      # prefill
        (128, 32, 8, 128),     # prefill
    ]

    for token_len, num_heads, num_kv_heads, head_dim in test_cases:
        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
        total_qkv_heads = num_heads + 2 * num_kv_heads
        x = torch.randn(token_len, total_qkv_heads, head_dim, device=device)
        positions = torch.randint(0, 1000, (token_len,), device=device, dtype=torch.int32)

        # warmup
        for _ in range(5):
            decode_apply_rope(x.clone(), cos_table, sin_table, positions, num_heads, num_kv_heads, head_dim)
            rope_qkv_decode(x.clone(), positions, cos_table, sin_table, num_kv_heads)
            rope_qkv_decode_smem(x.clone(), positions, cos_table, sin_table, num_kv_heads)
            apply_rope_qkv_ref(x, positions, cos_table, sin_table, num_kv_heads)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # rope0
        times = []
        for _ in range(num_iters):
            x_copy = x.clone()
            start.record()
            decode_apply_rope(x_copy, cos_table, sin_table, positions, num_heads, num_kv_heads, head_dim)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        rope0_ms = sum(times) / len(times)

        # rope_qkv_decode (float4)
        times = []
        for _ in range(num_iters):
            x_copy = x.clone()
            start.record()
            rope_qkv_decode(x_copy, positions, cos_table, sin_table, num_kv_heads)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        f4_ms = sum(times) / len(times)

        # rope_qkv_decode_smem
        times = []
        for _ in range(num_iters):
            x_copy = x.clone()
            start.record()
            rope_qkv_decode_smem(x_copy, positions, cos_table, sin_table, num_kv_heads)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        smem_ms = sum(times) / len(times)

        # torch reference
        times = []
        for _ in range(num_iters):
            start.record()
            apply_rope_qkv_ref(x, positions, cos_table, sin_table, num_kv_heads)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        torch_ms = sum(times) / len(times)

        config_str = f"[t={token_len},h={num_heads},kv={num_kv_heads},d={head_dim}]"
        print(f"  {config_str:>30s}  {rope0_ms:>10.3f}  {f4_ms:>10.3f}  {smem_ms:>12.3f}  {torch_ms:>10.3f}")


if __name__ == "__main__":
    test()
    test_performance()
