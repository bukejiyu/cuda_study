#include <torch/extension.h>

constexpr int kThreads = 256;

// ============================================================
// RoPE Decode Kernel (前后半配对, 无 smem)
// x:         [batch, num_heads, head_dim]   输入输出 inplace
// positions: [batch]                        每个序列当前 token 的绝对位置
// cos_table: [max_pos, half_dim]            预计算 cos
// sin_table: [max_pos, half_dim]            预计算 sin
//
// 线程映射: 每个 thread 处理一个 (batch, head, half_idx)
// 前后半配对: x[i] 和 x[half_dim+i] 配对，保证连续访存
// ============================================================
__global__ void rope_decode_kernel(
    float* __restrict__ x,
    const int* __restrict__ positions,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch, int num_heads, int half_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * half_dim;
    if (idx >= total) return;

    // 解码线性索引: idx = b * num_heads * half_dim + h * half_dim + i
    int i = idx % half_dim;
    idx /= half_dim;
    int h = idx % num_heads;
    int b = idx / num_heads;

    // 查表: 该 token 的绝对位置
    int pos = positions[b];
    float c = cos_table[pos * half_dim + i];
    float s = sin_table[pos * half_dim + i];

    // x 布局: [batch, num_heads, head_dim]，前后半配对
    int base = b * num_heads * (half_dim * 2) + h * (half_dim * 2);
    float x0 = x[base + i];                  // 前半
    float x1 = x[base + half_dim + i];       // 后半

    // 旋转: [x0, x1] 乘 2D 旋转矩阵
    x[base + i]             = x0 * c - x1 * s;
    x[base + half_dim + i]  = x0 * s + x1 * c;
}

// ============================================================
// RoPE QKV Decode Kernel (相邻配对 + float4 + GQA, 无 smem)
// x:         [batch, num_heads + 2*num_kv_heads, head_dim]  QKV fused
// positions: [batch]                                         每个 token 绝对位置
// cos_table: [max_pos, half_dim]                             预计算 cos
// sin_table: [max_pos, half_dim]                             预计算 sin
//
// QKV 布局: [Q0, Q1, ..., K0, K1, ..., V0, V1, ...]
//            [:, :num_heads, :]                               ← Q
//            [:, num_heads:num_heads+num_kv_heads, :]         ← K
//            [:, num_heads+num_kv_heads:, :]                  ← V (不处理)
//
// 只对 Q 和 K 做 RoPE, V 跳过
//
// 相邻配对 + float4:
//   每个 thread 用 float4 加载 4 个连续元素 [x[4t], x[4t+1], x[4t+2], x[4t+3]]
//   在 thread 内完成 2 个相邻 pair 的旋转:
//     pair 0: (x[4t],   x[4t+1]) → cos[2t],   sin[2t]
//     pair 1: (x[4t+2], x[4t+3]) → cos[2t+1], sin[2t+1]
//   float4 天然合并访存, 不需要 shared memory
//
// GQA: num_kv_heads < num_heads 时, K/V 的 head 数比 Q 少
//      例: LLaMA-2 70B  num_heads=64, num_kv_heads=8
//      MHA: num_kv_heads == num_heads, 退化为普通 QKV
//
// Block 映射: dim3(32, 4), 即 4 个 warp
//   threadIdx.x (0~31): head 内的 float4 索引, 32 threads = 1 warp = 1 head
//   threadIdx.y (0~3):  block 内的 QK head 编号
//   blockIdx.x: 覆盖 batch * ceil(total_qk_heads / 4) 个组
//   每个 warp 处理一个 (batch, qk_head), warp 内 32 threads 合并访存
// ============================================================
__global__ void rope_qkv_decode_kernel(
    float* __restrict__ x,
    const int* __restrict__ positions,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch, int num_heads, int num_kv_heads, int head_dim
) {
    int half_dim = head_dim / 2;
    int total_qk_heads = num_heads + num_kv_heads;

    // threadIdx.x = float4 索引 (0~31), threadIdx.y = block 内 head 编号 (0~3)
    int v = threadIdx.x;   // head 内第 v 个 float4
    int warp_local = threadIdx.y;  // block 内第几个 head

    // 全局 work index: blockIdx.x * 4 + threadIdx.y
    int work_idx = blockIdx.x * 4 + warp_local;
    if (work_idx >= batch * total_qk_heads) return;

    int b = work_idx / total_qk_heads;
    int qk_idx = work_idx % total_qk_heads;

    // 边界检查: float4 索引不能超出 half_dim
    if (v >= half_dim) return;  // head_dim=128 时 v∈[0,31], half_dim=64, 不会触发

    int pos = positions[b];
    int total_qkv_heads = num_heads + 2 * num_kv_heads;

    // Q 和 K 在 head 维度连续存储, head_offset = qk_idx
    int base = b * total_qkv_heads * head_dim + qk_idx * head_dim;

    // float4 加载 4 个连续元素
    float4 data = *reinterpret_cast<float4*>(x + base + v * 4);

    // float2 加载 cos/sin, 每个 thread 读连续 2 个值, warp 内 32 thread 覆盖 half_dim
    //   c.x=cos[2v], c.y=cos[2v+1]   s.x=sin[2v], s.y=sin[2v+1]
    float2 c = reinterpret_cast<const float2*>(cos_table)[pos * (half_dim / 2) + v];
    float2 s = reinterpret_cast<const float2*>(sin_table)[pos * (half_dim / 2) + v];

    // 相邻配对旋转:
    //   pair 0: (data.x, data.y) → c.x, s.x
    //   pair 1: (data.z, data.w) → c.y, s.y
    float4 out;
    out.x = data.x * c.x - data.y * s.x;
    out.y = data.x * s.x + data.y * c.x;
    out.z = data.z * c.y - data.w * s.y;
    out.w = data.z * s.y + data.w * c.y;

    *reinterpret_cast<float4*>(x + base + v * 4) = out;
}

// ============================================================
// RoPE QKV Decode Kernel (相邻配对 + shared memory + GQA)
// 与 float4 版本功能相同, 但用 smem 中转实现合并访存
// 适用于 head_dim 不是 4 的倍数的情况 (极少见)
// ============================================================
__global__ void rope_qkv_decode_smem_kernel(
    float* __restrict__ x,
    const int* __restrict__ positions,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int batch, int num_heads, int num_kv_heads, int head_dim
) {
    int half_dim = head_dim / 2;
    int tid = threadIdx.x;

    int total_qk_heads = num_heads + num_kv_heads;
    int b = blockIdx.x / total_qk_heads;
    int qk_idx = blockIdx.x % total_qk_heads;

    if (b >= batch || tid >= head_dim) return;

    int pos = positions[b];
    int total_qkv_heads = num_heads + 2 * num_kv_heads;
    int base = b * total_qkv_heads * head_dim + qk_idx * head_dim;

    extern __shared__ float smem[];

    // Step 1: 合并加载到 smem
    smem[tid] = x[base + tid];
    __syncthreads();

    // Step 2: 相邻配对旋转 (只需 half_dim 个线程)
    if (tid < half_dim) {
        float x0 = smem[2 * tid];
        float x1 = smem[2 * tid + 1];
        float c = cos_table[pos * half_dim + tid];
        float s = sin_table[pos * half_dim + tid];
        smem[2 * tid]     = x0 * c - x1 * s;
        smem[2 * tid + 1] = x0 * s + x1 * c;
    }
    __syncthreads();

    // Step 3: 合并写回
    x[base + tid] = smem[tid];
}

// ============================================================
// C++ 接口
// ============================================================
torch::Tensor rope_decode(torch::Tensor x, torch::Tensor positions,
                          torch::Tensor cos_table, torch::Tensor sin_table) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, num_heads, head_dim]");
    TORCH_CHECK(positions.dim() == 1, "positions must be 1D [batch]");
    TORCH_CHECK(cos_table.dim() == 2, "cos_table must be 2D [max_pos, half_dim]");
    TORCH_CHECK(sin_table.dim() == 2, "sin_table must be 2D [max_pos, half_dim]");

    int batch = x.size(0);
    int num_heads = x.size(1);
    int head_dim = x.size(2);
    int half_dim = head_dim / 2;

    int total = batch * num_heads * half_dim;
    int blocks = (total + kThreads - 1) / kThreads;

    rope_decode_kernel<<<blocks, kThreads>>>(
        x.data_ptr<float>(),
        positions.data_ptr<int>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        batch, num_heads, half_dim
    );

    return x;
}

// QKV fused + 相邻配对 + float4 + GQA
// x: [batch, num_heads + 2*num_kv_heads, head_dim]
torch::Tensor rope_qkv_decode(torch::Tensor x, torch::Tensor positions,
                               torch::Tensor cos_table, torch::Tensor sin_table,
                               int num_kv_heads) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, num_heads+2*num_kv_heads, head_dim]");

    int batch = x.size(0);
    int total_heads = x.size(1);
    int head_dim = x.size(2);
    int num_heads = total_heads - 2 * num_kv_heads;

    TORCH_CHECK(num_heads > 0, "invalid num_kv_heads for given x shape");
    TORCH_CHECK(total_heads == num_heads + 2 * num_kv_heads,
                "x.size(1) must equal num_heads + 2*num_kv_heads");
    TORCH_CHECK(head_dim % 4 == 0, "head_dim must be divisible by 4 for float4");

    int total_qk_heads = num_heads + num_kv_heads;
    // 每个 block 处理 4 个 QK head, grid 覆盖 ceil(batch*total_qk_heads / 4)
    int total_work = batch * total_qk_heads;
    int blocks = (total_work + 4 - 1) / 4;
    dim3 block_dim(32, 4);  // 4 个 warp, 每个 warp 处理 1 个 head

    rope_qkv_decode_kernel<<<blocks, block_dim>>>(
        x.data_ptr<float>(),
        positions.data_ptr<int>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        batch, num_heads, num_kv_heads, head_dim
    );

    return x;
}

// QKV fused + 相邻配对 + shared memory + GQA
// x: [batch, num_heads + 2*num_kv_heads, head_dim]
torch::Tensor rope_qkv_decode_smem(torch::Tensor x, torch::Tensor positions,
                                    torch::Tensor cos_table, torch::Tensor sin_table,
                                    int num_kv_heads) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, num_heads+2*num_kv_heads, head_dim]");

    int batch = x.size(0);
    int total_heads = x.size(1);
    int head_dim = x.size(2);
    int num_heads = total_heads - 2 * num_kv_heads;

    TORCH_CHECK(num_heads > 0, "invalid num_kv_heads for given x shape");
    TORCH_CHECK(total_heads == num_heads + 2 * num_kv_heads,
                "x.size(1) must equal num_heads + 2*num_kv_heads");

    int total_qk_heads = num_heads + num_kv_heads;
    int grid_size = batch * total_qk_heads;
    int smem_size = head_dim * sizeof(float);

    rope_qkv_decode_smem_kernel<<<grid_size, head_dim, smem_size>>>(
        x.data_ptr<float>(),
        positions.data_ptr<int>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        batch, num_heads, num_kv_heads, head_dim
    );

    return x;
}
