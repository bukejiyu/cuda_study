"""
prefill_apply_rope GQA 单测
测试目标: qkv_split_rope.cu 中 prefill_apply_rope kernel
- 输入: packed qkv [token_len, num_head+2*kv_num_head, head_dim]
- 输出: q [token_len, num_head, head_dim]
         k [token_len, kv_num_head, head_dim]
         v [token_len, kv_num_head, head_dim] (v 不做 rope)
- rope 方式: 相邻配对 (x[2i], x[2i+1]) 旋转
- 支持 varlen (variable-length sequences, packed format)
"""
import torch
from bh_ops import prefill_apply_rope


def build_rope_table(max_seq_len: int, head_dim: int, device="cuda"):
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    cos_table = torch.cos(freqs)  # [max_seq_len, head_dim/2]
    sin_table = torch.sin(freqs)
    # kernel 期望 shape: [1, max_seq_len, 1, head_dim/2]，reshape 后传入
    return cos_table.to(device), sin_table.to(device)


def ref_prefill_apply_rope(qkv, seq_lens, num_head, kv_num_head, cos_table, sin_table):
    """
    Python 参考实现（相邻配对 rope）
    qkv: [token_len, num_head+2*kv_num_head, head_dim]
    seq_lens: list[int], 每个 batch 的序列长度
    返回 q, k, v (split + rope applied)
    """
    token_len, total_head, head_dim = qkv.shape
    q_out = torch.zeros(token_len, num_head, head_dim, device=qkv.device)
    k_out = torch.zeros(token_len, kv_num_head, head_dim, device=qkv.device)
    v_out = torch.zeros(token_len, kv_num_head, head_dim, device=qkv.device)

    token_offset = 0
    for seq_len in seq_lens:
        for s in range(seq_len):
            t = token_offset + s
            cos = cos_table[s]  # [head_dim/2]
            sin = sin_table[s]

            # Q
            for h in range(num_head):
                x = qkv[t, h].clone()
                x_even = x[0::2]
                x_odd  = x[1::2]
                q_out[t, h, 0::2] = x_even * cos - x_odd * sin
                q_out[t, h, 1::2] = x_even * sin + x_odd * cos

            # K
            for h in range(kv_num_head):
                x = qkv[t, num_head + h].clone()
                x_even = x[0::2]
                x_odd  = x[1::2]
                k_out[t, h, 0::2] = x_even * cos - x_odd * sin
                k_out[t, h, 1::2] = x_even * sin + x_odd * cos

            # V: 不做 rope
            v_out[t] = qkv[t, num_head + kv_num_head:]

        token_offset += seq_len

    return q_out, k_out, v_out


def make_varlen_inputs(seq_lens, num_head, kv_num_head, head_dim, device="cuda"):
    token_len = sum(seq_lens)
    bs = len(seq_lens)

    qkv = torch.randn(token_len, num_head + 2 * kv_num_head, head_dim, device=device)

    # batch_per_token[i] = 第 i 个 token 属于哪个 batch
    batch_per_token = torch.zeros(token_len, dtype=torch.int32, device=device)
    offset = 0
    for b, slen in enumerate(seq_lens):
        batch_per_token[offset:offset + slen] = b
        offset += slen

    # cu_seq_len: [bs+1], cumsum of seq_lens starting from 0
    cu_seq_len = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    for i, slen in enumerate(seq_lens):
        cu_seq_len[i + 1] = cu_seq_len[i] + slen

    return qkv, batch_per_token, cu_seq_len


def run_test(seq_lens, num_head, kv_num_head, head_dim=128, atol=1e-4):
    device = "cuda"
    max_seq_len = max(seq_lens) + 10

    cos_table, sin_table = build_rope_table(max_seq_len, head_dim, device)
    # kernel 期望 [1, max_seq_len, 1, head_dim/2]
    cos_emb = cos_table.unsqueeze(0).unsqueeze(2).contiguous()  # [1,max_seq_len,1,head_dim/2]
    sin_emb = sin_table.unsqueeze(0).unsqueeze(2).contiguous()

    qkv, batch_per_token, cu_seq_len = make_varlen_inputs(seq_lens, num_head, kv_num_head, head_dim, device)

    # kernel 调用
    q_cuda, k_cuda, v_cuda = prefill_apply_rope(
        qkv, cos_emb, sin_emb, batch_per_token, cu_seq_len, num_head, kv_num_head, head_dim
    )

    # 参考实现
    q_ref, k_ref, v_ref = ref_prefill_apply_rope(qkv, seq_lens, num_head, kv_num_head, cos_table, sin_table)

    q_ok = torch.allclose(q_cuda, q_ref, atol=atol)
    k_ok = torch.allclose(k_cuda, k_ref, atol=atol)
    v_ok = torch.allclose(v_cuda, v_ref, atol=atol)

    config = f"seq_lens={seq_lens}, h={num_head}, kv={kv_num_head}, d={head_dim}"
    if q_ok and k_ok and v_ok:
        print(f"  PASS  {config}")
    else:
        errs = []
        if not q_ok:
            errs.append(f"q_max_err={(q_cuda-q_ref).abs().max():.2e}")
        if not k_ok:
            errs.append(f"k_max_err={(k_cuda-k_ref).abs().max():.2e}")
        if not v_ok:
            errs.append(f"v_max_err={(v_cuda-v_ref).abs().max():.2e}")
        print(f"  FAIL  {config}  {' '.join(errs)}")

    return q_ok and k_ok and v_ok


def test_correctness():
    print("=" * 60)
    print("prefill_apply_rope GQA Correctness Tests")
    print("=" * 60)

    cases = [
        # (seq_lens, num_head, kv_num_head)
        ([4],              8,  8),   # 单 batch MHA
        ([4],              8,  2),   # 单 batch GQA 4x
        ([2, 3],           8,  2),   # 多 batch GQA
        ([1, 4, 2],        16, 4),   # 3 batch GQA 4x
        ([8, 7, 6, 5],     32, 8),   # LLaMA-2 style GQA
        ([1],              8,  2),   # 极小序列
        ([64],             32, 8),   # 长序列
    ]

    all_pass = True
    for seq_lens, num_head, kv_num_head in cases:
        ok = run_test(seq_lens, num_head, kv_num_head)
        if not ok:
            all_pass = False

    print()
    print("All PASSED!" if all_pass else "Some FAILED!")


if __name__ == "__main__":
    test_correctness()
