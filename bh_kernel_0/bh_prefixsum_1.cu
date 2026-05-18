#include <torch/extension.h>
constexpr int kPerGroupSize =1024;
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kElemt = 4;
constexpr int kWaprNum = kThreads/kWarpSize;

__device__ float warp_prefix_sum(float data){
    for(int offset=1;offset<32;offset=offset*2){
        float tmp = __shfl_up_sync(0xFFFFFFFF, data, offset);
        data += (threadIdx.x%32 >= offset) ? tmp : 0.0f;
    }
    return data;
}
__global__ void per_group_prefix_sum(float* input,float* output,float* groups_sum,int n,int group_size){
    int warp_id = threadIdx.x / kWarpSize;
    __shared__ float warp_smem[kWaprNum];
    __shared__ float thread_smem[kThreads];
    __shared__ float pre_thread_group_acc;
    int lane_id = threadIdx.x % kWarpSize;
    // [BUG FIX 8] 移除了这里的 pre_thread_group_acc = 0.0f 初始化
    // 改到每个 group 循环内部重置，否则跨 group 时上一个 group 的累加值会泄漏到下一个 group
    // if (threadIdx.x ==0) pre_thread_group_acc = 0.0f;
    // __syncthreads();

    // [BUG FIX 1] blockDim.x -> blockIdx.x
    // blockDim.x 是线程数(128)，不是 block 索引，导致 group 0~127 完全没被处理
    // for(int gi = blockDim.x;gi < group_size;gi += gridDim.x){
    for(int gi = blockIdx.x; gi < group_size; gi += gridDim.x){
        // [BUG FIX 8] 每个 group 开始时重置 pre_thread_group_acc
        if (threadIdx.x == 0) pre_thread_group_acc = 0.0f;
        __syncthreads();

        int g_base = gi * kPerGroupSize;
        int num_iter;
        // [BUG FIX 2] num_iter 计算要考虑每个线程处理 kElemt=4 个元素
        // 原来除以 kThreads，但实际上每轮迭代 kThreads 个线程 * kElemt 个元素 = 512 个元素
        // 对于 kPerGroupSize=1024，原来 num_iter=8，实际应该为 2
        if (g_base + kPerGroupSize <= n){
            // num_iter = (kPerGroupSize + kThreads - 1)/ kThreads;
            num_iter = (kPerGroupSize + kThreads * kElemt - 1) / (kThreads * kElemt);
        }else{
            // num_iter = (n - g_base + kThreads - 1)/ kThreads;
            num_iter = (n - g_base + kThreads * kElemt - 1) / (kThreads * kElemt);
        }

        // [BUG FIX 3 & 4] 循环结构重写
        // 原来用 i=threadIdx.x 作为循环变量，混淆了线程索引和迭代序号
        // 导致只有 threadIdx.x < num_iter 的少数线程参与（num_iter 很小如 2）
        // 正确做法：所有 128 个线程都参与每次迭代，用 iter 遍历迭代次数
        // for(int i = threadIdx.x;i < num_iter;i += kThreads){
        for(int iter = 0; iter < num_iter; iter++){
            int global_index = g_base + iter * kThreads * kElemt + threadIdx.x * kElemt;
            float acc = 0.0f;
            float vec[kElemt] = {0};
            // [BUG FIX 5] < n 改为 <= n（非越界问题，是性能问题）
            // 当 global_index + kElemt == n 时，4 个元素恰好都在合法范围内 [global_index, n-1]
            // 用 < 会导致本来可以向量化的路径走了慢速 fallback，结果一样但更慢
            // if(global_index+kElemt<n){
            if(global_index+kElemt<=n){
                float4 tmp = *reinterpret_cast<float4*>(&input[global_index]);
                vec[0] = tmp.x;
                vec[1] = vec[0]+tmp.y;
                vec[2] = vec[1]+tmp.z;
                vec[3] = vec[2]+tmp.w;
                acc = vec[3];
            }else{
                for(int j = 0;j < kElemt; j++){
                    float tmp = global_index + j < n ? input[global_index + j]:0.0f;
                    acc = acc + tmp;
                    vec[j] = acc;
                    // if(global_index+j<n){
                    //     float tmp_bias = j>0? vec[j-1]:0.0f;
                    //     vec[j] = input[global_index+j]+tmp_bias;
                    //     if((global_index+j)==n-1) acc = vec[j];
                    // }
                }
            }
            acc = warp_prefix_sum(acc);
            thread_smem[threadIdx.x] = acc;
            if(lane_id==31){
                warp_smem[warp_id] = acc;
            }
            __syncthreads();
            if(warp_id==0){
                float tmp =  lane_id<kWaprNum?warp_smem[lane_id]:0.0f;
                tmp = warp_prefix_sum(tmp);
                if(lane_id<kWaprNum) warp_smem[lane_id] = tmp;
            }
            __syncthreads();

            // [BUG FIX 6] threadIdx.x>0 -> lane_id>0
            // 原来用 threadIdx.x>0 判断，但 warp 1 的首线程(threadIdx.x=32)：
            //   thread_bias = thread_smem[31] = sum_warp_0（整个 warp0 的总和）
            //   warp_bias   = warp_smem[0]    = sum_warp_0（也是 warp0 的总和）
            //   total = 2 * sum_warp_0  ← 重复计算！
            // 正确做法：lane_id==0 时 thread_bias=0，由 warp_bias 独自承担跨 warp 偏移
            // float thread_bias  = threadIdx.x>0? thread_smem[threadIdx.x-1]:0.0f;
            float thread_bias  = lane_id > 0 ? thread_smem[threadIdx.x-1] : 0.0f;
            float warp_bias = warp_id > 0 ? warp_smem[warp_id-1]:0.0f;
            float pre_acc = pre_thread_group_acc;
            __syncthreads();

            for(int j=0;j<kElemt;j++){
                vec[j] += thread_bias+warp_bias+pre_acc;
            }
            // [BUG FIX 5] 同上，<n 改为 <=n
            // if(global_index+kElemt<n){
            if(global_index+kElemt<=n){
                float4 tmp;
                tmp.x = vec[0];
                tmp.y = vec[1];
                tmp.z = vec[2];
                tmp.w = vec[3];
                *reinterpret_cast<float4*>(&output[global_index]) = tmp;
            }else{
                for(int j=0;j<kElemt;j++){
                    if(global_index+j<n){
                        output[global_index+j] = vec[j];
                    }
                }
            }
            if(threadIdx.x==0) pre_thread_group_acc += warp_smem[kWaprNum-1];
            __syncthreads();
        }
        if (threadIdx.x==0) groups_sum[gi] = pre_thread_group_acc;
        __syncthreads();
    }
}
__global__ void prefixsum_group(float* groups_sum,int group_size){
    int tid = threadIdx.x;
    int lane_id = tid % kWarpSize;
    int warp_id = tid / kWarpSize;
    // [BUG FIX 2] 同 per_group_prefix_sum，num_iter 要除以 kThreads*kElemt
    // int num_iter = (group_size + kThreads -1)/kThreads;
    int num_iter = (group_size + kThreads * kElemt - 1) / (kThreads * kElemt);
    __shared__ float pre_iter_sum;
    __shared__ float thread_smem[kThreads];
    __shared__ float warp_smem[kThreads/kWarpSize];
    if(tid == 0) pre_iter_sum = 0.0f;
    __syncthreads();

    // [BUG FIX 3 & 4] 同 per_group_prefix_sum，循环结构重写
    // 原来用 i=tid 作为循环变量+元素索引，混淆了线程索引和迭代序号
    // 导致线程间读取的 groups_sum 数据重叠（相邻线程读重叠的 float4）
    // for(int i=tid;i<num_iter;i+=kThreads){
    for(int iter = 0; iter < num_iter; iter++){
        int idx = iter * kThreads * kElemt + tid * kElemt;
        float vec[kElemt] = {0};
        float acc=0.0f;
        // [BUG FIX 5] < 改为 <=
        // if(i+kElemt<group_size){
        if(idx+kElemt<=group_size){
            float4 tmp = *reinterpret_cast<float4*>(&groups_sum[idx]);
            vec[0] = tmp.x;
            vec[1] = vec[0]+tmp.y;
            vec[2] = vec[1]+tmp.z;
            vec[3] = vec[2]+tmp.w;
            acc = vec[3];
        }else{
            for(int j=0;j<kElemt;j++){
                float tmp = idx+j < group_size ? groups_sum[idx+j]:0.0f;
                acc = acc + tmp;
                vec[j] = acc;
                // if(idx+j<group_size){
                //     float tmp_bias = j>0? vec[j-1]:0.0f;
                //     vec[j] = groups_sum[idx+j]+tmp_bias;
                //     if((idx+j) == group_size-1 ) acc = vec[j];
                // }
            }
        }
        acc = warp_prefix_sum(acc);
        thread_smem[tid] = acc;
        if(lane_id==31) warp_smem[warp_id] = acc;
        __syncthreads();
        if(warp_id==0){
            float tmp = lane_id<kWaprNum? warp_smem[lane_id]:0.0f;
            tmp = warp_prefix_sum(tmp);
            if(lane_id<kWaprNum) warp_smem[lane_id] = tmp;
        }
        __syncthreads();
        // [BUG FIX 6] tid>0 -> lane_id>0，同 per_group_prefix_sum 的原因
        // float thread_bias = tid>0? thread_smem[tid-1]:0.0f;
        float thread_bias = lane_id > 0 ? thread_smem[tid-1] : 0.0f;
        float warp_bias = warp_id>0? warp_smem[warp_id-1]:0.0f;
        float pre_iter = pre_iter_sum;
        __syncthreads();

        // [BUG FIX 5] < 改为 <=
        // if(i+kElemt<group_size){
        if(idx+kElemt<=group_size){
            float4 tmp;
            tmp.x = vec[0]+thread_bias+warp_bias+pre_iter;
            tmp.y = vec[1]+thread_bias+warp_bias+pre_iter;
            tmp.z = vec[2]+thread_bias+warp_bias+pre_iter;
            tmp.w = vec[3]+thread_bias+warp_bias+pre_iter;
            *reinterpret_cast<float4*>(&groups_sum[idx]) = tmp;
        }else{
            for(int j=0;j<kElemt;j++){
                if(idx+j<group_size){
                    groups_sum[idx+j] = vec[j]+thread_bias+warp_bias+pre_iter;
                }
            }
        }
        if(tid==0) pre_iter_sum += warp_smem[kWaprNum-1];
        __syncthreads();
    }
}
__global__ void output_prefixsum(float* output,float* groups_sum,int n,int group_size){
    int tid=threadIdx.x;
    for(int gid = blockIdx.x;gid<group_size;gid+=gridDim.x){
        // [BUG FIX 7] 完全重写内层循环
        // 原来的 for(int i=tid;i<n;i+=kThreads) 遍历了整个数组 n 个元素
        // 导致每个 block 把 group_acc 加到所有元素上，而不是只加到自己的 group 内
        // 这造成：(1) 元素被多个 block 重复累加 group_acc (2) 不同 block 写同一位置存在竞争
        // 正确做法：只遍历当前 group 的元素
        int group_start = gid * kPerGroupSize;
        int group_end = min(group_start + kPerGroupSize, n);
        int group_len = group_end - group_start;
        float group_acc = gid>0?groups_sum[gid-1]:0.0f;

        int num_vec = (group_len + kElemt - 1) / kElemt;
        for(int vi = tid; vi < num_vec; vi += kThreads){
            int global_index = group_start + vi * kElemt;
            if (global_index + kElemt <= group_end){
                float4 tmp = *reinterpret_cast<float4*>(&output[global_index]);
                tmp.x += group_acc;
                tmp.y += group_acc;
                tmp.z += group_acc;
                tmp.w += group_acc;
                *reinterpret_cast<float4*>(&output[global_index]) = tmp;
            }else{
                for(int j=0;j<kElemt;j++){
                    if(global_index+j < group_end){
                        output[global_index+j] += group_acc;
                    }
                }
            }
        }
        // 原来的错误代码（保留参考）:
        // for(int i=tid;i<n;i+=kThreads){
        //     int global_index = gid*kPerGroupSize + i;
        //     float group_acc = gid>0?groups_sum[gid-1]:0.0f;
        //     if (global_index + kElemt<n){
        //         float4 tmp = *reinterpret_cast<float4*>(&output[global_index]);
        //         tmp.x = tmp.x + group_acc;
        //         tmp.y = tmp.y + group_acc;
        //         tmp.z = tmp.z + group_acc;
        //         tmp.w = tmp.w + group_acc;
        //         *reinterpret_cast<float4*>(&output[global_index]) = tmp;
        //     }else{
        //         for(int j=0;j<kElemt;j++){
        //             if((global_index+j)<n)  output[global_index+j] += group_acc;
        //         }
        //     }
        // }
    }

}
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scan_then_fan_debug(torch::Tensor input) {
//     TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
//     TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

//     auto shape = input.sizes();
//     auto flat = input.reshape(-1);
//     size_t n = flat.size(0);
//     if (n == 0) {
//         auto empty = input.clone();
//         return std::make_tuple(empty, empty, empty);
//     }

//     torch::Tensor output = torch::zeros_like(flat);
//     int group_size = (n + kPerGroupSize -1)/kPerGroupSize;
//     auto groups_sum = torch::zeros({group_size}, input.options());
//     dim3 grid_dims;
//     grid_dims.x = min(group_size,65535);
//     per_group_prefix_sum<<<grid_dims,kThreads>>>(flat.data_ptr<float>(),output.data_ptr<float>(),groups_sum.data_ptr<float>(),n,group_size);

//     auto groups_sum_stage1 = groups_sum.clone();

//     prefixsum_group<<<1,kThreads>>>(groups_sum.data_ptr<float>(),group_size);

//     auto groups_sum_stage2 = groups_sum.clone();

//     output_prefixsum<<<grid_dims,kThreads>>>(output.data_ptr<float>(),groups_sum.data_ptr<float>(),n,group_size);
//     return std::make_tuple(output, groups_sum_stage1, groups_sum_stage2);
// }

torch::Tensor scan_then_fan(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto shape = input.sizes();
    auto flat = input.reshape(-1);
    size_t n = flat.size(0);
    if (n == 0) {
        return input.clone();
    }

    torch::Tensor output = torch::zeros_like(flat);
    int group_size = (n + kPerGroupSize -1)/kPerGroupSize;
    auto groups_sum = torch::zeros({group_size}, input.options());
    dim3 grid_dims;
    grid_dims.x = min(group_size,65535);
    per_group_prefix_sum<<<grid_dims,kThreads>>>(flat.data_ptr<float>(),output.data_ptr<float>(),groups_sum.data_ptr<float>(),n,group_size);
    prefixsum_group<<<1,kThreads>>>(groups_sum.data_ptr<float>(),group_size);
    output_prefixsum<<<grid_dims,kThreads>>>(output.data_ptr<float>(),groups_sum.data_ptr<float>(),n,group_size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_then_fan", &scan_then_fan, "Scan-then-fan prefix sum");
    // m.def("scan_then_fan_debug", &scan_then_fan_debug, "Debug scan-then-fan prefix sum");
}
