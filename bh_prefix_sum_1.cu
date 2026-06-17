#include <torch/extension.h>

constexpr int kElemsPerThread = 4;
constexpr int kThreadNum = 256;
constexpr int kPartSize = kThreadNum * kElemsPerThread;  // 1024

// ============================================================
// Warp 内前缀和（寄存器级，__shfl_up_sync）
// ============================================================
__device__ float ScanWarp(float val) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += tmp;
        }
    }
    return val;
}

// ============================================================
// Block 内前缀和（warp scan + warp 级 scan）
// 256 threads = 8 warps
// ============================================================
//整体思路 
//1.32个线程 每个warp 通过 __shfl_up_sync 进行前缀和
//2.通过warp_smem 存 每个warp的 最大前缀和
//3.计算warp_smem的 warp的前缀和
//4.warp_id>0的部分 warp_id -1 的最大前缀和
//5.最后将计算好的整个block的前缀和 更新到 主smem中

__device__ void ScanBlock(float* shm) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    __shared__ float warp_smem[8];  // 256 / 32 = 8 warps

    float val = shm[tid];
    float warp_result = ScanWarp(val);

    if (lane_id == 31) {
        warp_smem[warp_id] = warp_result;
    }
    __syncthreads();

    if (warp_id == 0) {
        float warp_val = warp_smem[lane_id];
        warp_smem[lane_id] = ScanWarp(warp_val);
    }
    __syncthreads();

    if (warp_id > 0) {
        warp_result += warp_smem[warp_id - 1];
    }

    shm[tid] = warp_result;
    __syncthreads();
}

// ============================================================
// Kernel 1: 每个 block 做局部前缀和
//   - 每线程读 4 个元素，寄存器内串行前缀和
//   - ScanBlock 对线程级总和做并行前缀和
//   - 写 output（局部前缀和）+ part_sum（块总和 B）
// ============================================================
__global__ void LocalScanKernel(const float* __restrict__ input,
                                float* __restrict__ part_sum,
                                float* __restrict__ output,
                                size_t n, size_t part_num) {
    for (int part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        int base = part_i * kPartSize;

        // 每个线程读 4 个元素，寄存器内串行前缀和
        float local_scan[kElemsPerThread];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kElemsPerThread; ++j) {
            int idx = base + threadIdx.x * kElemsPerThread + j;
            float val = idx < n ? input[idx] : 0.0f;
            acc += val;
            local_scan[j] = acc;
        }

        // ScanBlock 对每个线程的局部总和做并行前缀和
        __shared__ float smem[kThreadNum];
        smem[threadIdx.x] = acc;
        __syncthreads();
        ScanBlock(smem);

        // smem[tid] = 这个线程及之前所有线程的累加和
        // acc = 这个线程自己的局部总和
        // 差值 = 这个线程之前的所有线程的总和 = 偏移
        float thread_offset = smem[threadIdx.x] - acc;

#pragma unroll
        for (int j = 0; j < kElemsPerThread; ++j) {
            //每个线程去处理4个
            int idx = base + threadIdx.x * kElemsPerThread + j;
            if (idx < n) {
                output[idx] = local_scan[j] + thread_offset;
            }
        }

        // 块总和写入 part_sum（即 B 数组）
        if (threadIdx.x == 0) {
            part_sum[part_i] = smem[kThreadNum - 1];
        }
    }
}

// ============================================================
// Kernel 2: 对 part_sum（B 数组）做前缀和
//   - stride 循环 + running base_sum，支持任意 part_num
//   - 不再限制 part_num <= 1024
// ============================================================
__global__ void ScanPartSumKernel(float* __restrict__ part_sum,
                                  size_t part_num) {
    __shared__ float smem[kThreadNum];
    __shared__ float base_sum;

    if (threadIdx.x == 0) base_sum = 0.0f;
    __syncthreads();

    for (size_t i = threadIdx.x; i < part_num; i += blockDim.x) {
        smem[threadIdx.x] = part_sum[i];
        __syncthreads();
        ScanBlock(smem);
        __syncthreads();

        part_sum[i] = smem[threadIdx.x] + base_sum;

        if (threadIdx.x == blockDim.x - 1) {
            base_sum += smem[threadIdx.x];
        }
        __syncthreads();
    }
}

// ============================================================
// Kernel 3: 每个块加上前面所有块的总和偏移
//   - part_sum[part_i - 1] 只读一次到 shared memory，广播
// ============================================================
__global__ void AddBaseOffsetKernel(float* __restrict__ part_sum,
                                    float* __restrict__ output,
                                    size_t n, size_t part_num) {
    __shared__ float offset;

    for (int part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        if (part_i == 0) continue;

        if (threadIdx.x == 0) {
            offset = part_sum[part_i - 1];
        }
        __syncthreads();

        float off = offset;
#pragma unroll
        for (int j = 0; j < kElemsPerThread; ++j) {
            int idx = part_i * kPartSize + threadIdx.x * kElemsPerThread + j;
            if (idx < n) {
                output[idx] += off;
            }
        }
    }
}

// ============================================================
// PyTorch 接口
// ============================================================
torch::Tensor scan_then_fan(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto shape = input.sizes();
    auto flat = input.reshape(-1);
    size_t n = flat.size(0);
    if (n == 0) {
        return input.clone();
    }

    size_t part_num = (n + kPartSize - 1) / kPartSize;
    size_t block_num = std::min<size_t>(part_num, 65535);

    auto output = torch::empty_like(flat);
    auto part_sum = torch::empty({(int64_t)part_num}, flat.options());

    // Step 1: 每个 block 做局部前缀和 → output + part_sum(B)
    LocalScanKernel<<<block_num, kThreadNum>>>(
        flat.data_ptr<float>(), part_sum.data_ptr<float>(),
        output.data_ptr<float>(), n, part_num);

    // Step 2: 对 B 做前缀和
    ScanPartSumKernel<<<1, kThreadNum>>>(
        part_sum.data_ptr<float>(), part_num);

    // Step 3: 加上 B 的偏移
    AddBaseOffsetKernel<<<block_num, kThreadNum>>>(
        part_sum.data_ptr<float>(), output.data_ptr<float>(), n, part_num);

    return output.reshape(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_then_fan", &scan_then_fan, "Scan-then-fan prefix sum");
}
