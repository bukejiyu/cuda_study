#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// 方法1: Kogge-Stone Inclusive Scan block内，只能保证在 某个特定值内 prefixsum
// ============================================================
template <int BLOCK_SIZE>
__global__ void kogge_stone_scan_kernel(const float* input, float* output, int D) {
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int row = blockIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + tid;
    int gid = row * D + col;

    s_data[tid] = (col < D) ? input[gid] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        float val = 0.0f;
        if (tid >= offset) val = s_data[tid - offset];
        __syncthreads();
        s_data[tid] += val;
        __syncthreads();
    }

    if (col < D) output[gid] = s_data[tid];
}

// ============================================================
// 方法2: Blelloch Exclusive Scan
// ============================================================
template <int BLOCK_SIZE>
__global__ void blelloch_scan_kernel(const float* input, float* output, int D) {
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int row = blockIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + tid;
    int gid = row * D + col;

    s_data[tid] = (col < D) ? input[gid] : 0.0f;
    __syncthreads();

    // 上扫
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < BLOCK_SIZE && index >= offset)
            s_data[index] += s_data[index - offset];
        __syncthreads();
    }

    if (tid == 0) s_data[BLOCK_SIZE - 1] = 0.0f;
    __syncthreads();

    // 下扫
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < BLOCK_SIZE && index >= offset) {
            float tmp = s_data[index - offset];
            s_data[index - offset] = s_data[index];
            s_data[index] += tmp;
        }
        __syncthreads();
    }

    if (col < D) output[gid] = s_data[tid];
}

// ============================================================
// 方法3: Three-Phase 多Block扫描
// ============================================================
template <int BLOCK_SIZE>
__global__ void scan_block_kernel(const float* input, float* output,
                                   float* block_sums, int D, int num_blocks) {
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int row = blockIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + tid;
    int gid = row * D + col;

    s_data[tid] = (col < D) ? input[gid] : 0.0f;
    __syncthreads();

    // offset = 1  [a0,a0+a1][a2,a2+a3]....
    // offset = 2  [a0,a0+a1,a2,a0+a1+a2+a3] ....
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < BLOCK_SIZE && index >= offset)
            s_data[index] += s_data[index - offset];
        __syncthreads();
    }
    // 这仅填充了 每个 block的 最后一个 值
    if (tid == 0) block_sums[row * num_blocks + blockIdx.x] = s_data[BLOCK_SIZE - 1];
    // reset 将每个block的最后一个值置0 
    if (tid == 0) s_data[BLOCK_SIZE - 1] = 0.0f;
    __syncthreads();

    // 下扫例子
    //以len=8 为例子 root [a0,a0+a1,a2,a0+...+a3,a4,a4+a5,a6,0]
    //offset = 4  父节点 index[7], 这个算法 右节点=父节点 左节点=index[3] 进行交换分发 并加到父节点 
    //结果 即 root[3] = root[7] root[7]= root[7]+root[3]
    // root [a0,a0+a1,a2,0,a4,a4+a5,a6,a0+...+a3]

    //offset = 2  [ 父节点 root[7] 左节点 root[5] ] [ 父节点 root[3]  左节点 root[1]]
    //root [a0 , 0 , a2, a0+a1 ,a4, a0+...+a3,a6,a0+...+a5]

    //offset =1 [父节点 root[7] 左节点 root[6]] .... [父节点[1] 左节点[0]]
    //root [0,a0,a0+a1,a0+...+a2,a0+...+a3,a0+...+a4,a0+...+a5,a0+...+a6]
    
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        int index = (tid + 1) * offset * 2 - 1; //还是拿每个 chunk的最右边？
        if (index < BLOCK_SIZE && index >= offset) {
            float tmp = s_data[index - offset];
            s_data[index - offset] = s_data[index];
            s_data[index] += tmp;
        }
        __syncthreads();
    }

    if (col < D) output[gid] = s_data[tid];
}

template <int BLOCK_SIZE>
__global__ void scan_block_sums_kernel(float* block_sums, int num_blocks) {
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int row = blockIdx.y;
    int idx = row * num_blocks + tid;

    s_data[tid] = (tid < num_blocks) ? block_sums[idx] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < BLOCK_SIZE && index >= offset)
            s_data[index] += s_data[index - offset];
        __syncthreads();
    }
    if (tid == 0) s_data[BLOCK_SIZE - 1] = 0.0f;
    __syncthreads();

    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < BLOCK_SIZE && index >= offset) {
            float tmp = s_data[index - offset];
            s_data[index - offset] = s_data[index];
            s_data[index] += tmp;
        }
        __syncthreads();
    }

    if (tid < num_blocks) block_sums[idx] = s_data[tid];
}

__global__ void add_block_sums_kernel(float* output, const float* block_sums, int D, int num_blocks) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int gid = row * D + col;
    if (col < D) output[gid] += block_sums[row * num_blocks + blockIdx.x];
}

// ============================================================
// Torch 接口
// ============================================================

// Kogge-Stone inclusive scan: input [N, D] -> output [N, D]，沿最后一维做 inclusive scan
torch::Tensor kogge_stone_scan(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    constexpr int BLOCK_SIZE = 1024;

    auto output = torch::empty_like(input);
    int num_blocks = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_blocks, N);

    kogge_stone_scan_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), D);

    return output;
}

// Blelloch exclusive scan: input [N, D] -> output [N, D]，沿最后一维做 exclusive scan
torch::Tensor blelloch_scan(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    constexpr int BLOCK_SIZE = 1024;

    auto output = torch::empty_like(input);
    int num_blocks = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_blocks, N);

    blelloch_scan_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), D);

    return output;
}

// Three-Phase exclusive scan: input [N, D] -> output [N, D]，支持 D > BLOCK_SIZE
torch::Tensor three_phase_scan(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    constexpr int BLOCK_SIZE = 1024;

    auto output = torch::empty_like(input);
    int num_blocks = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_blocks, N);

    // block_sums: [N, num_blocks]
    auto block_sums = torch::empty({N, num_blocks}, input.options());

    // Phase 1: 块内 scan + 记录块总和
    scan_block_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        block_sums.data_ptr<float>(), D, num_blocks);

    // Phase 2 & 3: 块间扫描 + 加回偏移
    if (num_blocks > 1) {
        // 对每行的 block_sums 做 scan
        dim3 grid2(1, N);
        scan_block_sums_kernel<BLOCK_SIZE><<<grid2, BLOCK_SIZE>>>(
            block_sums.data_ptr<float>(), num_blocks);

        // 加回偏移
        add_block_sums_kernel<<<grid, BLOCK_SIZE>>>(
            output.data_ptr<float>(), block_sums.data_ptr<float>(), D, num_blocks);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kogge_stone_scan", &kogge_stone_scan, "Kogge-Stone inclusive scan");
    m.def("blelloch_scan", &blelloch_scan, "Blelloch exclusive scan");
    m.def("three_phase_scan", &three_phase_scan, "Three-Phase exclusive scan");
}
