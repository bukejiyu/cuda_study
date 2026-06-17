#include <torch/extension.h>

// 声明各 op 的 C++ 接口
torch::Tensor reduce_sum(torch::Tensor input);
torch::Tensor softmax(torch::Tensor input);
torch::Tensor transpose(torch::Tensor input);
torch::Tensor scan_then_fan(torch::Tensor input);
torch::Tensor online_softmax(torch::Tensor input);
torch::Tensor online_softmax_v2(torch::Tensor input);
torch::Tensor transpose_v2(torch::Tensor input);
torch::Tensor transpose_v3(torch::Tensor input);
torch::Tensor group_norm(torch::Tensor input, int num_groups,
                         c10::optional<torch::Tensor> gamma,
                         c10::optional<torch::Tensor> beta);
torch::Tensor group_norm_v2(torch::Tensor input, int num_groups,
                         c10::optional<torch::Tensor> gamma,
                         c10::optional<torch::Tensor> beta);
torch::Tensor rope_decode(torch::Tensor x, torch::Tensor positions,
                          torch::Tensor cos_table, torch::Tensor sin_table);
torch::Tensor rope_qkv_decode(torch::Tensor x, torch::Tensor positions,
                               torch::Tensor cos_table, torch::Tensor sin_table,
                               int num_kv_heads);
torch::Tensor rope_qkv_decode_smem(torch::Tensor x, torch::Tensor positions,
                                    torch::Tensor cos_table, torch::Tensor sin_table,
                                    int num_kv_heads);
torch::Tensor decode_apply_rope(torch::Tensor qkv, torch::Tensor cos_emb, torch::Tensor sin_emb,
                                torch::Tensor positions, int num_head, int kv_num_head, int head_dim);
std::vector<torch::Tensor> prefill_apply_rope(torch::Tensor qkv, torch::Tensor cos_emb, torch::Tensor sin_emb,
                                 torch::Tensor batch_per_token, torch::Tensor cu_seq_len,
                                 int num_head, int kv_num_head, int head_dim);
std::vector<torch::Tensor> qkv_split_rope(torch::Tensor qkv, torch::Tensor cos_emb, torch::Tensor sin_emb,
                                 torch::Tensor batch_per_token, torch::Tensor cu_seq_len,
                                 int num_head, int kv_num_head, int head_dim);
torch::Tensor navie_gemm(torch::Tensor A, torch::Tensor B);
torch::Tensor smem_gemm(torch::Tensor A, torch::Tensor B);
torch::Tensor smem_t8x8_gemm(torch::Tensor A, torch::Tensor B);
std::vector<torch::Tensor> block_wise_fp8_quant(torch::Tensor x, int block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce sum along last dim (CUDA)");
    m.def("softmax", &softmax, "Softmax (CUDA)");
    m.def("transpose", &transpose, "Transpose (CUDA)");
    m.def("transpose_v2", &transpose_v2, "Transpose (CUDA)");
    m.def("scan_then_fan", &scan_then_fan, "Scan-then-fan prefix sum");
    m.def("online_softmax", &online_softmax, "online_softmax");
    m.def("online_softmax_v2", &online_softmax_v2, "online_softmax_v2");
    m.def("transpose_v3",&transpose_v3,"transpose_v3");
    m.def("group_norm", &group_norm, "Group Normalization (CUDA)");
    m.def("group_norm_v2", &group_norm_v2, "Group Normalization v2 (CUDA)");
    m.def("rope_decode", &rope_decode, "RoPE Decode (CUDA)");
    m.def("rope_qkv_decode", &rope_qkv_decode, "RoPE QKV Decode float4 (CUDA)");
    m.def("rope_qkv_decode_smem", &rope_qkv_decode_smem, "RoPE QKV Decode smem (CUDA)");
    m.def("decode_apply_rope", &decode_apply_rope, "RoPE QKV Decode v0 (CUDA)");
    m.def("prefill_apply_rope", &prefill_apply_rope, "RoPE Prefill continuous batch (CUDA)");
    m.def("qkv_split_rope", &qkv_split_rope, "QKV split + RoPE prefill (CUDA)");
    m.def("navie_gemm", &navie_gemm, "Naive GEMM fp32 (CUDA)");
    m.def("smem_gemm", &smem_gemm, "Shared Memory GEMM fp32 (CUDA)");
    m.def("smem_t8x8_gemm", &smem_t8x8_gemm, "Shared Memory Tile 8x8 GEMM fp32x4 (CUDA)");
    m.def("block_wise_fp8_quant", &block_wise_fp8_quant, "Block-wise FP8 E4M3 quantization (CUDA)");
}
