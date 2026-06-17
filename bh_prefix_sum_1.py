import torch


def scan_then_fan(input_tensor: torch.Tensor) -> torch.Tensor:
    """Scan-then-fan prefix sum (inclusive scan).

    Logic:
      1. Split input into parts, compute local prefix sum within each part,
         and record the sum of each part.
      2. Compute prefix sum over the part sums.
      3. Add the cumulative part offset to every element (except part 0).

    Args:
        input_tensor: 1-D float32 tensor of shape (n,).

    Returns:
        1-D float32 tensor of shape (n,) — the inclusive prefix sum.
    """
    assert input_tensor.ndim == 1, "Only 1-D input is supported"
    n = input_tensor.shape[0]
    if n == 0:
        return input_tensor.clone()

    device = input_tensor.device
    part_size = 1024  # tuned, same as the CUDA kernel

    # ---- Step 1: local scan within each part ----
    # Pad input to a multiple of part_size, then reshape into (part_num, part_size)
    part_num = (n + part_size - 1) // part_size
    padded = torch.nn.functional.pad(input_tensor, (0, part_num * part_size - n))
    parts = padded.reshape(part_num, part_size)

    # Local inclusive prefix sum within each part
    local_scan = parts.cumsum(dim=1)

    # Sum of each part (will be used as the "base offset" in step 3)
    part_sums = local_scan[:, -1]  # shape: (part_num,)

    # ---- Step 2: prefix sum over part sums ----
    # After this, part_sums[i] = sum of part_0 + part_1 + ... + part_i
    part_sums = part_sums.cumsum(dim=0)

    # ---- Step 3: add the base offset from previous parts ----
    # For part_i > 0, every element in that part needs to add the cumulative
    # sum of all previous parts (i.e., part_sums[part_i - 1]).
    # Part 0 needs no offset (same as the CUDA kernel's `if part_i == 0: continue`).
    offsets = torch.cat([torch.zeros(1, device=device, dtype=input_tensor.dtype),
                         part_sums[:-1]])
    output = (local_scan + offsets.unsqueeze(1)).reshape(-1)

    # Trim the padding and return
    return output[:n].to(input_tensor.dtype)


if __name__ == "__main__":
    # Simple correctness test
    for n in [1, 10, 1023, 1024, 1025, 5000, 10000]:
        x = torch.rand(n, dtype=torch.float32)
        expected = x.cumsum(0)
        result = scan_then_fan(x)
        assert torch.allclose(result, expected, atol=1e-5), f"Failed for n={n}"
    print("All tests passed!")
