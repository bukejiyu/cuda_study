import torch
from bh_ops import block_wise_fp8_quant

MAX_FP8 = 448.0


def ref_block_wise_fp8(x: torch.Tensor, block_size: int):
    """Python reference: per-block (128元素/block) fp8 量化"""
    token_num, hidden_size = x.shape
    assert hidden_size % block_size == 0
    num_blocks = hidden_size // block_size

    x_fp32 = x.float()
    x_reshaped = x_fp32.view(token_num, num_blocks, block_size)  # [T, B, block_size]

    scale = x_reshaped.abs().max(dim=-1).values  # [T, B]
    scale = scale.clamp(min=1e-10)
    scale_factor = scale / MAX_FP8  # [T, B]

    # 量化：先缩放到 fp8 范围，再 cast 到 fp8，再 dequant 回 float 用于误差对比
    x_scaled = x_reshaped / scale_factor.unsqueeze(-1) * MAX_FP8 / MAX_FP8  # = x / scale_factor
    x_quant_fp8 = x_scaled.to(torch.float8_e4m3fn)

    return x_quant_fp8.view(token_num, hidden_size), scale_factor


def dequant(x_quant, x_scale, block_size):
    """dequant: x_quant (fp8) * scale_factor -> float"""
    token_num, hidden_size = x_quant.shape
    num_blocks = hidden_size // block_size
    x_f = x_quant.float().view(token_num, num_blocks, block_size)
    return (x_f * x_scale.unsqueeze(-1)).view(token_num, hidden_size)


def test_correctness():
    device = "cuda"
    block_size = 128
    print("=" * 60)
    print("Block-wise FP8 Quantization Correctness Tests")
    print("=" * 60)

    cases = [
        (1, 128),
        (4, 256),
        (8, 512),
        (16, 1024),
        (32, 2048),
    ]

    all_pass = True
    for token_num, hidden_size in cases:
        x = torch.randn(token_num, hidden_size, device=device)

        # kernel 输出
        x_quant, x_scale = block_wise_fp8_quant(x, block_size)

        # reference
        ref_quant, ref_scale = ref_block_wise_fp8(x, block_size)

        # scale 对比
        scale_ok = torch.allclose(x_scale, ref_scale, rtol=1e-4, atol=1e-6)

        # dequant 后与原始值误差（fp8 精度有限，允许较大 atol）
        x_dequant = dequant(x_quant, x_scale, block_size)
        ref_dequant = dequant(ref_quant, ref_scale, block_size)
        dequant_ok = torch.allclose(x_dequant, ref_dequant, atol=0.05)

        ok = scale_ok and dequant_ok
        if not ok:
            print(f"  [T={token_num}, H={hidden_size}]: FAIL  scale_ok={scale_ok}  dequant_ok={dequant_ok}")
            all_pass = False
        else:
            max_err = (x_dequant - ref_dequant).abs().max().item()
            print(f"  [T={token_num}, H={hidden_size}]: PASS  max_err={max_err:.4f}")

    # 边界：全零
    x = torch.zeros(4, 128, device=device)
    x_quant, x_scale = block_wise_fp8_quant(x, block_size)
    # scale 应夹紧到 epsilon/MAX_FP8
    ok = (x_quant.float().abs().max().item() == 0.0)
    print(f"  [all zeros]: {'PASS' if ok else 'FAIL'}")

    # 边界：极大值，scale 不应 overflow
    x = torch.full((4, 128), 1e4, device=device)
    x_quant, x_scale = block_wise_fp8_quant(x, block_size)
    ok = x_quant.float().isfinite().all().item()
    print(f"  [large values 1e4]: {'PASS' if ok else 'FAIL'}")

    print()
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")


if __name__ == "__main__":
    test_correctness()
