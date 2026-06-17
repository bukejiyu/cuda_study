import torch

from torch.utils.cpp_extension import load

prefix_sum = load(
    name="prefix_sum",
    sources=["bh_prefix_sum.cu"],
    verbose=True,
)

def test(name, fn, input_tensor, scan_type="exclusive"):
    output = fn(input_tensor)
    if scan_type == "exclusive":
        ref = torch.cumsum(input_tensor, dim=-1).roll(1, dims=-1)
        ref[..., 0] = 0
    else:
        ref = torch.cumsum(input_tensor, dim=-1)

    match = torch.allclose(output, ref, atol=1e-3)
    max_err = (output - ref).abs().max().item()
    print(f"[{name}] {input_tensor.shape} {'PASS' if match else 'FAIL'}  max_err={max_err:.6f}")
    if not match:
        idx = (output - ref).abs().argmax().item()
        row = idx // input_tensor.shape[1]
        col = idx % input_tensor.shape[1]
        print(f"  first mismatch at [{row},{col}]: got {output[row,col].item()}, expected {ref[row,col].item()}")
    return match

all_pass = True

# 测试1: D < BLOCK_SIZE
x = torch.randn(4, 512, device="cuda")
all_pass &= test("Blelloch D<BLOCK_SIZE", prefix_sum.blelloch_scan, x)
all_pass &= test("Kogge-Stone D<BLOCK_SIZE", prefix_sum.kogge_stone_scan, x, "inclusive")
all_pass &= test("Three-Phase D<BLOCK_SIZE", prefix_sum.three_phase_scan, x)

# 测试2: D = BLOCK_SIZE
x = torch.randn(4, 1024, device="cuda")
all_pass &= test("Blelloch D=BLOCK_SIZE", prefix_sum.blelloch_scan, x)
all_pass &= test("Kogge-Stone D=BLOCK_SIZE", prefix_sum.kogge_stone_scan, x, "inclusive")
all_pass &= test("Three-Phase D=BLOCK_SIZE", prefix_sum.three_phase_scan, x)

# 测试3: D > BLOCK_SIZE (关键: 多block)
x = torch.randn(4, 2048, device="cuda")
all_pass &= test("Blelloch D>BLOCK_SIZE", prefix_sum.blelloch_scan, x)
all_pass &= test("Kogge-Stone D>BLOCK_SIZE", prefix_sum.kogge_stone_scan, x, "inclusive")
all_pass &= test("Three-Phase D>BLOCK_SIZE", prefix_sum.three_phase_scan, x)

# 测试4: D 不是 BLOCK_SIZE 的整数倍
x = torch.randn(8, 1500, device="cuda")
all_pass &= test("Blelloch D=1500", prefix_sum.blelloch_scan, x)
all_pass &= test("Three-Phase D=1500", prefix_sum.three_phase_scan, x)

# 测试5: D 远大于 BLOCK_SIZE
x = torch.randn(2, 4096, device="cuda")
all_pass &= test("Three-Phase D=4096", prefix_sum.three_phase_scan, x)

# 测试6: N=1
x = torch.randn(1, 2048, device="cuda")
all_pass &= test("Three-Phase N=1", prefix_sum.three_phase_scan, x)

print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
