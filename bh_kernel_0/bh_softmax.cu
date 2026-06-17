#include <torch/extension.h>
constexpr int kThreads = 128;
__device__ float warpMax(float data){
    for(int offset=16;offset>0;offset=offset/2){
        float tmp = __shfl_xor_sync(0xFFFFFFFF,data,offset);
        data = fmaxf(data,tmp);
    }
    return data;
}
__device__ float reducesum(float data){
    for(int offset=16;offset>0;offset=offset/2){
        float tmp = __shfl_xor_sync(0xFFFFFFFF,data,offset);
        data += tmp;
    }
    return data;
}
__global__ void softmax_kernel(float* input,float* output,int N,int D){
    int tid = threadIdx.x;
    int lane_id = tid%32;
    int warp_id = tid/32;
    __shared__ float max_smem[kThreads/32];
    __shared__ float sum_smem[kThreads/32];
    __shared__ float max_all;
    __shared__ float sum_all;
    constexpr int num_warp = kThreads/32;
    int bid = blockIdx.x;
    if (tid ==0){
        max_all = -FLT_MAX;
        sum_all = 0.0f;
    }
    __syncthreads();
    int num_iter = (D + kThreads - 1)/kThreads;
    for(int iter=0;iter<num_iter;iter++){
        int base_offset = bid*D;
        int dim_offset = iter*kThreads + tid;
        float data = dim_offset < D ? input[base_offset+dim_offset] : -FLT_MAX;
        data = warpMax(data);
        if(lane_id==0) max_smem[warp_id] = data;
        __syncthreads();
        if(warp_id == 0){
            float tmp =  lane_id < num_warp ? max_smem[lane_id] : -FLT_MAX;
            tmp = warpMax(tmp);
            if(lane_id<num_warp)  max_smem[lane_id] = tmp;
        }
        __syncthreads();
        float thread_max = max_smem[0];
        float max_all_now = max_all;
        __syncthreads();
        if(tid == 0) max_all = fmaxf(max_all_now,thread_max);
        __syncthreads();
    }

    float final_max = max_all;
    __syncthreads();

    for(int iter=0;iter<num_iter;iter++){
        int base_offset = bid*D;
        int dim_offset = iter*kThreads + tid;
        float data = dim_offset < D ? expf(input[base_offset+dim_offset] - final_max):0.0f;
        data = reducesum(data);
        if(lane_id==0) sum_smem[warp_id] = data;
        __syncthreads();
        if(warp_id == 0){
            float tmp =  lane_id < num_warp ? sum_smem[lane_id] : 0.0f;
            tmp = reducesum(tmp);
            if(lane_id<num_warp)  sum_smem[lane_id] = tmp;
        }
        __syncthreads();
        float thread_all = sum_smem[0];
        float sum_all_now = sum_all;
        __syncthreads();
        if(tid == 0) sum_all = sum_all_now + thread_all;
        __syncthreads();
    }

    float final_sum = sum_all;
    __syncthreads();

    for(int iter=0;iter<num_iter;iter++){
        int base_offset = bid*D;
        int dim_offset = iter*kThreads + tid;
        if(dim_offset<D){
            output[base_offset+dim_offset] =  expf(input[base_offset+dim_offset]-final_max)/final_sum;
        }
    }
}

//softmax 需要的是 max和 exp（x-max）的sum
//1.找局部max
__global__ void online_softmax_kernel(float* input,float* output,int N,int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid%32;
    int warp_id = tid/32;
    constexpr int num_warps = kThreads/32;
    int num_iter = (D + kThreads - 1)/kThreads;
    
    __shared__ float pre_max;
    __shared__ float pre_sum;

    __shared__ float max_seme[num_warps];
    __shared__ float sum_smem[num_warps];
    
    if(tid == 0){
        pre_max = -FLT_MAX;
        pre_sum = 0.0f;
    }

    for(int iter = 0 ; iter < num_iter ; iter++){
        // float max = -FLT_MAX;
        int base_offset = bid * D;
        int dim_offset = iter * kThreads + tid;
        float max = dim_offset < D ? input[base_offset + dim_offset] : -FLT_MAX;
        max = warpMax(max);
        //局部max
        float exp_val = dim_offset < D ? expf(input[base_offset + dim_offset]-max) : 0.0f;
        float data = reducesum(exp_val);
        if(lane_id==0) {
            max_seme[warp_id]= max;
            sum_smem[warp_id] = data;
        }
        __syncthreads();
        //先算局部的 max和 sum
        if(warp_id == 0){
            float tmp_sum = lane_id<num_warps?sum_smem[lane_id]:0.0f;
            float tmp_max = lane_id<num_warps?max_seme[lane_id]:-FLT_MAX;
            float new_max = warpMax(tmp_max);
            //更新新的max
            tmp_sum = tmp_sum*expf(tmp_max-new_max);
            tmp_sum = reducesum(tmp_sum);
            if(lane_id<num_warps) {
                max_seme[lane_id] = new_max;
                sum_smem[lane_id] = tmp_sum;
            }    
        }
        __syncthreads();
        //这时候更新每个 iter的 sum和 max
        float thread_max = max_seme[0];
        float thread_sum = sum_smem[0];
        float pre_iter_max = pre_max;
        float pre_iter_sum = pre_sum;
        __syncthreads();
        float new_max = fmaxf(pre_iter_max,thread_max);
        pre_iter_sum = pre_iter_sum*expf(pre_iter_max-new_max);
        thread_sum = thread_sum*expf(thread_max-new_max);

        if(tid==0){
            pre_max = new_max;
            pre_sum = pre_iter_sum + thread_sum;
        }
        __syncthreads();    
    }
    for(int iter = 0 ; iter < num_iter ; iter++){
        int base_offset = bid * D;
        int dim_offset = iter * kThreads + tid;
        float final_max = pre_max;
        float final_sum = pre_sum;
        __syncthreads();
        if(dim_offset<D){
            output[base_offset+dim_offset] = expf(input[base_offset+dim_offset]-final_max)/final_sum;
        }
    }
}

// ============================================================
// online_softmax_v2: 线程级 online 累积 + float4 向量化
// ============================================================
constexpr int kV2Threads = 256;
constexpr int kV2Warps = kV2Threads / 32;
constexpr int kVecSize = 4;  // 每个线程每次处理 4 个 float (float4)

__global__ void online_softmax_v2_kernel(float* input, float* output, int N, int D) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    __shared__ float s_max[kV2Warps];
    __shared__ float s_sum[kV2Warps];
    __shared__ float g_max;
    __shared__ float g_sum;

    int base = bid * D;
    int elems_per_iter = kV2Threads * kVecSize;
    int num_iter = (D + elems_per_iter - 1) / elems_per_iter;

    // ===== Pass 1: 每个线程独立做 online 累积 =====
    // 关键：循环内没有任何 syncthreads，每个线程各自维护 running max/sum
    float m = -FLT_MAX;  // 线程级 running max
    float d = 0.0f;      // 线程级 running sum

    for (int i = 0; i < num_iter; i++) {
        int offset = i * elems_per_iter + tid * kVecSize;

        // 主路径：float4 向量化加载（需要对齐且不越界）
        bool aligned = ((base + offset) % kVecSize == 0);
        if (aligned && offset + kVecSize <= D) {
            float4 data = *reinterpret_cast<float4*>(input + base + offset);

            // 对 4 个元素依次做 online 更新：
            //   m_new = max(m_old, x)
            //   d_new = d_old * exp(m_old - m_new) + exp(x - m_new)
            #pragma unroll
            for (int j = 0; j < kVecSize; j++) {
                float val = (&data.x)[j];
                float old_m = m;
                m = fmaxf(m, val);
                d = d * expf(old_m - m) + expf(val - m);
            }
        } else {
            // 边界回退：标量逐个处理
            for (int j = 0; j < kVecSize; j++) {
                int idx = offset + j;
                if (idx < D) {
                    float val = input[base + idx];
                    float old_m = m;
                    m = fmaxf(m, val);
                    d = d * expf(old_m - m) + expf(val - m);
                }
            }
        }
    }

    // ===== Block Reduce: 找全局 max =====
    float w_max = warpMax(m);
    if (lane_id == 0) s_max[warp_id] = w_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = lane_id < kV2Warps ? s_max[lane_id] : -FLT_MAX;
        g_max = warpMax(v);
    }
    __syncthreads();

    // 修正：将每个线程的 sum 从 "基于线程 local max" 校正到 "基于全局 max"
    d *= expf(m - g_max);

    // ===== Block Reduce: 求全局 sum =====
    float w_sum = reducesum(d);
    if (lane_id == 0) s_sum[warp_id] = w_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = lane_id < kV2Warps ? s_sum[lane_id] : 0.0f;
        g_sum = reducesum(v);
    }
    __syncthreads();

    // ===== Pass 2: 写出结果 =====
    for (int i = 0; i < num_iter; i++) {
        int offset = i * elems_per_iter + tid * kVecSize;

        bool aligned = ((base + offset) % kVecSize == 0);
        if (aligned && offset + kVecSize <= D) {
            float4 data = *reinterpret_cast<float4*>(input + base + offset);
            float4 out;
            out.x = expf(data.x - g_max) / g_sum;
            out.y = expf(data.y - g_max) / g_sum;
            out.z = expf(data.z - g_max) / g_sum;
            out.w = expf(data.w - g_max) / g_sum;
            *reinterpret_cast<float4*>(output + base + offset) = out;
        } else {
            for (int j = 0; j < kVecSize; j++) {
                int idx = offset + j;
                if (idx < D) {
                    output[base + idx] = expf(input[base + idx] - g_max) / g_sum;
                }
            }
        }
    }
}

torch::Tensor softmax(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    torch::Tensor output = torch::zeros_like(input);
    size_t N = input.size(0);
    size_t D = input.size(1);
    dim3 gird_dim;
    gird_dim.x = N;
    softmax_kernel<<<gird_dim,kThreads>>>(input.data_ptr<float>(),output.data_ptr<float>(),N,D);
    return output;
}

torch::Tensor online_softmax(torch::Tensor input){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    torch::Tensor output = torch::zeros_like(input);
    size_t N = input.size(0);
    size_t D = input.size(1);
    dim3 gird_dim;
    gird_dim.x = N;
    online_softmax_kernel<<<gird_dim,kThreads>>>(input.data_ptr<float>(),output.data_ptr<float>(),N,D);
    return output;
}

torch::Tensor online_softmax_v2(torch::Tensor input){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    torch::Tensor output = torch::zeros_like(input);
    size_t N = input.size(0);
    size_t D = input.size(1);
    dim3 grid_dim;
    grid_dim.x = N;
    online_softmax_v2_kernel<<<grid_dim,kV2Threads>>>(input.data_ptr<float>(),output.data_ptr<float>(),N,D);
    return output;
}

