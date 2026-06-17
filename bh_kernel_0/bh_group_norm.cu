#include <torch/extension.h>

constexpr int kThreads = 256;
__device__ float warp_reducesum(float data){
    for(int offset=16;offset>0;offset=offset/2){
        data += __shfl_xor_sync(0xFFFFFFFF,data,offset);
    }
    return data;
}
// 每个 block 处理一个 (n, group)：对该组内 G*H*W 个元素求 mean/var，再归一化
__global__ void group_norm_kernel(const float *input, float *output,
                                  const float *gamma, const float *beta,
                                  int N, int C, int H, int W, int num_groups) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;  // bid = n * num_groups + g
    int warp_id = tid/32;
    int lane_id = tid%32;
    constexpr int num_warp = kThreads/32;

    int n = bid / num_groups;
    int g = bid % num_groups;

    int G = C / num_groups;  // 每组通道数
    int group_size = G * H * W;  // 每组元素总数

    // 该组在 input 中的起始偏移: 第 n 个样本, 第 g 组的第一个通道
    int group_offset = n * C * H * W + g * G * H * W;

    // ---------- 第1步: 求 sum 和 sum_sq ----------
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // stride loop: 每个 thread 处理多个元素
    for (int i = tid; i < group_size; i += kThreads) {
        float val = input[group_offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    local_sum = warp_reducesum(local_sum);
    local_sum_sq = warp_reducesum(local_sum_sq);
    // shared memory 做 block 内 reduction
    __shared__ float s_sum[num_warp];
    __shared__ float s_sum_sq[num_warp];

    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
    __syncthreads();
    if(warp_id ==0){
        float tmp_s_sum = lane_id<num_warp? s_sum[lane_id]:0.0f;
        float tmp_s_sum_sq = lane_id<num_warp? s_sum_sq[lane_id]:0.0f;
        tmp_s_sum=warp_reducesum(tmp_s_sum);
        tmp_s_sum_sq=warp_reducesum(tmp_s_sum_sq);
        if(lane_id<num_warp){
            s_sum[lane_id] = tmp_s_sum;
            s_sum_sq[lane_id] = tmp_s_sum_sq;
        }
    }
    __syncthreads();

    // reduce sum
    // for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    //     if (tid < stride) {
    //         s_sum[tid] += s_sum[tid + stride];
    //         s_sum_sq[tid] += s_sum_sq[tid + stride];
    //     }
    //     __syncthreads();
    // }
    float g_sum = s_sum[0];
    float g_sum_2 = s_sum_sq[0];
    __syncthreads();

    float mean = g_sum / group_size;
    float var = g_sum_2 / group_size - mean * mean;

    // ---------- 第2步: 归一化 + 仿射变换 ----------
    float inv_std = rsqrtf(var + 1e-5f);

    for (int i = tid; i < group_size; i += kThreads) {
        int c_in_group = i / (H * W);  // 组内通道索引 [0, G)
        int c_global = g * G + c_in_group;  // 全局通道索引

        float val = input[group_offset + i];
        float x_norm = (val - mean) * inv_std;

        // 仿射变换: gamma[c] * x_norm + beta[c]
        if (gamma != nullptr && beta != nullptr) {
            x_norm = gamma[c_global] * x_norm + beta[c_global];
        }
        output[group_offset + i] = x_norm;
    }
}

torch::Tensor group_norm(torch::Tensor input, int num_groups,
                         c10::optional<torch::Tensor> gamma_opt,
                         c10::optional<torch::Tensor> beta_opt) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (NCHW)");
    TORCH_CHECK(input.size(1) % num_groups == 0, "C must be divisible by num_groups");

    const float *gamma_ptr = gamma_opt.has_value() ? gamma_opt->data_ptr<float>() : nullptr;
    const float *beta_ptr  = beta_opt.has_value()  ? beta_opt->data_ptr<float>()  : nullptr;

    torch::Tensor output = torch::zeros_like(input);
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // 一个 block 处理一个 (n, group)
    int blocks = N * num_groups;
    group_norm_kernel<<<blocks, kThreads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        gamma_ptr, beta_ptr,
        N, C, H, W, num_groups);

    return output;
}
