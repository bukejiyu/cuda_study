import torch
from bh_ops import prefill_apply_rope

def build_rope_table(max_seq_len, head_dim, device="cuda"):
    half_dim = head_dim // 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    cos_table = torch.cos(freqs)
    sin_table = torch.sin(freqs)
    return cos_table.to(device), sin_table.to(device)

def apply_rope_qkv_ref_packed(x, batch_per_token, cu_seq_len, cos_table, sin_table, num_kv_heads):
    """
    相邻配对 QKV RoPE 参考实现（连续打包版本）。
    x: [token_len, num_heads + 2*num_kv_heads, head_dim]
    batch_per_token: [token_len]
    cu_seq_len: [bs+1]
    """
    token_len = x.size(0)
    num_heads = x.size(1) - 2 * num_kv_heads
    head_dim = x.size(2)
    half_dim = head_dim // 2

    out = x.clone()

    for t in range(token_len):
        bs = batch_per_token[t].item()
        seq_start = cu_seq_len[bs].item()
        pos_in_seq = t - seq_start  # 当前 token 在序列内的位置

        # 对 Q 和 K 做 RoPE
        for qk_idx in range(num_heads + num_kv_heads):
            cos = cos_table[pos_in_seq]  # [half_dim]
            sin = sin_table[pos_in_seq]

            head_data = out[t, qk_idx, :]  # [head_dim]
            x_even = head_data[0::2]  # [half_dim]
            x_odd  = head_data[1::2]

            rotated = torch.empty_like(head_data)
            rotated[0::2] = x_even * cos - x_odd * sin
            rotated[1::2] = x_even * sin + x_odd * cos
            out[t, qk_idx, :] = rotated

    # V 不处理
    return out

def test():
    max_seq_len = 512
    device = "cuda"

    # 测试用例: 多个不同长度的序列打包在一起
    # (seq_lengths, num_heads, num_kv_heads, head_dim)
    test_cases = [
        ([3, 2, 4], 8, 8, 128),       # 3个序列, MHA
        ([5, 3], 16, 4, 128),          # 2个序列, GQA
        ([1, 1, 1, 1], 32, 8, 128),    # 4个decode序列
        ([10], 32, 32, 128),           # 单序列
        ([2, 6, 3, 1], 32, 8, 128),   # 4个不同长度序列, GQA
    ]

    all_pass = True
    for seq_lengths, num_heads, num_kv_heads, head_dim in test_cases:
        bs = len(seq_lengths)
        token_len = sum(seq_lengths)
        total_qkv_heads = num_heads + 2 * num_kv_heads

        cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)

        # 构造 cu_seq_len
        cu_seq_len = [0]
        for l in seq_lengths:
            cu_seq_len.append(cu_seq_len[-1] + l)
        cu_seq_len_tensor = torch.tensor(cu_seq_len, device=device, dtype=torch.int32)

        # 构造 batch_per_token
        batch_per_token = []
        for bs_id, seq_len in enumerate(seq_lengths):
            batch_per_token.extend([bs_id] * seq_len)
        batch_per_token_tensor = torch.tensor(batch_per_token, device=device, dtype=torch.int32)

        # 随机输入
        x = torch.randn(token_len, total_qkv_heads, head_dim, device=device)

        # CUDA kernel
        x_copy = x.clone()
        out_cuda = prefill_apply_rope(x_copy, cos_table, sin_table, batch_per_token_tensor, cu_seq_len_tensor, num_heads, num_kv_heads, head_dim)

        # 参考实现
        out_ref = apply_rope_qkv_ref_packed(x, batch_per_token_tensor, cu_seq_len_tensor, cos_table, sin_table, num_kv_heads)

        # 检查精度
        ok = torch.allclose(out_cuda, out_ref, atol=1e-5)
        max_err = (out_cuda - out_ref).abs().max().item()

        # 检查 V 没被修改
        v_start = num_heads + num_kv_heads
        v_end = num_heads + 2 * num_kv_heads
        v_ok = torch.allclose(out_cuda[:, v_start:v_end, :], x[:, v_start:v_end, :], atol=1e-7)

        status = "PASS" if (ok and v_ok) else "FAIL"
        if not ok or not v_ok:
            all_pass = False
        v_status = "" if v_ok else " [V modified!]"
        seq_str = str(seq_lengths)

        print(f"  seqs={seq_str}, h={num_heads}, kv={num_kv_heads}, d={head_dim}: {status}  max_err={max_err:.2e}{v_status}")

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")

if __name__ == "__main__":
    test()
