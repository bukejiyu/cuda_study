#include <torch/extension.h>
constexpr const int kThreadNum = 128;
constexpr const int kElemst = 4;
constexpr const int perBlockNum = 1024;
constexpr const int kWarpSize = 32;

__device__ float warp_prefix_sum(float data){
    for(int offset=1;offset<kWarpSize;offset=offset*2){
        float tmp = __shfl_up_sync(0xFFFFFFFF,data,offset);
        data += threadIdx.x%32>=offset?tmp:0.0f;
    }
    return data;
}

__global__ void pre_block_prefix_sum(float* input ,float* output,float* blocks,int block_nums,int n){
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int lane_id = tid%32;
    __shared__ float pre_acc;
    __shared__ float block_smem[kThreadNum];
    // [BUG FIX] 原始代码: block_prefix_sum 是 __device__ 函数，内部声明 __shared__ float block_smem_1[4]
    // 原因: __device__ 函数内的 __shared__ 变量被编译器 inline 后，与外部 kernel 的 pre_acc、block_smem
    //       在 shared memory 中可能重叠，导致 block_prefix_sum 内部的写操作与外部对 pre_acc 的读操作
    //       产生不可预测的 WAR/WAW 竞争（compute-sanitizer racecheck 报告 0x410 地址处 400+ hazards）
    // 修正: 将 block_prefix_sum 内联展开，__shared__ 变量统一声明在 kernel 顶层
    __shared__ float warp_smem[kThreadNum/kWarpSize];

    for(int block_id = blockIdx.x;block_id < block_nums;block_id += gridDim.x){
        if(tid==0) pre_acc = 0.0f;
        __syncthreads();
        constexpr int num_iters = (perBlockNum/kElemst + kThreadNum - 1) / kThreadNum;
        for(int iter = 0;iter < num_iters;iter++){
            int i = iter * kThreadNum + tid;
            float acc = 0.0f;
            float vec[kElemst];
            int base = block_id*perBlockNum + i*kElemst;
            if(base + kElemst<=n){
                float4 tmp = *reinterpret_cast<float4*>(&input[base]);
                acc = tmp.x+tmp.y+tmp.z+tmp.w;
                vec[0] = tmp.x;
                vec[1] = tmp.x+tmp.y;
                vec[2] = tmp.x+tmp.y+tmp.z;
                vec[3] = tmp.x+tmp.y+tmp.z+tmp.w;
            }else{
                for(int j = 0;j < kElemst;j++){
                    float tmp = base + j < n ? input[base + j]:0.0f;
                    acc = acc + tmp;
                    vec[j] = acc;
                }
            }

            // --- inline block_prefix_sum (原始代码是 __device__ 函数调用) ---
            block_smem[tid] = acc;
            // [BUG FIX] 原始代码: block_smem[tid] = acc; 后直接调用 block_prefix_sum(block_smem);
            // 原因: block_prefix_sum 内部第一行 float val = smem[tid] 读取 block_smem,
            //       但所有线程写 block_smem[tid] = acc 和读取之间没有 __syncthreads(),
            //       虽然 tid 只读自己写的位置(寄存器级别可见), 但编译器优化可能重排,
            //       且 inline 后与 warp_smem/pre_acc 的 shared memory 布局冲突加剧了问题
            // 修正: 显式加 __syncthreads() 保证写完再读
            __syncthreads();
            float val = block_smem[tid];
            val = warp_prefix_sum(val);
            if(lane_id==31){
                warp_smem[warp_id] = val;
            }
            __syncthreads();
            if(warp_id==0){
                float v2 = lane_id<kThreadNum/kWarpSize?warp_smem[lane_id]:0.0f;
                v2 = warp_prefix_sum(v2);
                if(lane_id<kThreadNum/kWarpSize)
                    warp_smem[lane_id] = v2;
            }
            __syncthreads();
            block_smem[tid] = val + (warp_id>0?warp_smem[warp_id-1]:0.0f);
            __syncthreads();
            // --- end inline block_prefix_sum ---

            float bias = tid>0?block_smem[tid-1]:0.0f;

            // [BUG FIX] 原始代码:
            //   for(int j=0;j<kElemst;j++){
            //       if(base+j<n) output[base+j] = vec[j] + bias + pre_acc;
            //   }
            //   if(tid==0) pre_acc += block_smem[kThreadNum-1];
            //   __syncthreads();
            //
            // 原因: 所有线程读 pre_acc (shared memory) 和 thread 0 写 pre_acc 之间
            //       没有 __syncthreads() 隔离。warp 0 (含 thread 0) 可能先跑完 output 写回,
            //       直接进入 pre_acc += ... 更新 shared memory, 而其他 warp 还在读旧的 pre_acc。
            //       compute-sanitizer racecheck 报告 WAR hazard at __shared__ 0x410 (pre_acc),
            //       Read Thread 127 vs Write Thread 0, 共 400+ hazards。
            //
            // 修正: 先将 pre_acc 读到每个线程的寄存器 cur_pre_acc, 加 __syncthreads()
            //       确保所有线程读完后再由 thread 0 更新
            float cur_pre_acc = pre_acc;
            __syncthreads();
            if(base+kElemst<=n){
                float4 tmp_out;
                tmp_out.x = vec[0] + bias + cur_pre_acc;
                tmp_out.y = vec[1] + bias + cur_pre_acc;
                tmp_out.z = vec[2] + bias + cur_pre_acc;
                tmp_out.w = vec[3] + bias + cur_pre_acc;
                *reinterpret_cast<float4*>(&output[base]) = tmp_out;
            }else{
                for(int j = 0;j < kElemst;j++){
                    if (base + j < n)
                        output[base+j] = vec[j] + bias + cur_pre_acc;
                }
            }
            if(tid==0){
                pre_acc += block_smem[kThreadNum-1];
            }
            __syncthreads();
        }

        // [BUG FIX] 原始代码: blocks[block_id] = pre_acc;
        // 原因: 128 个线程同时写同一个全局内存地址 (WAW), 虽然值相同,
        //       但在 NVIDIA 文档中属于未定义行为, 可能在某些硬件上产生问题
        // 修正: 只有 tid==0 写
        if(tid==0) blocks[block_id] = pre_acc;
        // [BUG FIX] 原始代码: 此处没有 __syncthreads()
        // 原因: 外层 for 循环处理多个 data block 时, 下一轮迭代 tid==0 写 pre_acc=0.0f
        //       可能覆盖其他线程还在读的 blocks[block_id]=pre_acc 中的 pre_acc
        // 修正: 加 __syncthreads() 确保所有线程完成当前迭代后再进入下一轮
        __syncthreads();
    }
}

__global__ void prefix_sum_blocks(float* blocks,int block_nums){
    int tid=threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    __shared__ float smem[kThreadNum/kWarpSize];
    __shared__ float pre_acc;
    // [BUG FIX] 原始代码: pre_acc = 0.0f; (所有线程都写,是WAW竞争)
    // 修正: 只有tid==0写,避免多线程同时写shared memory
    if(tid==0) pre_acc = 0.0f;
    __syncthreads();
    int num_iters = (block_nums + kThreadNum - 1) / kThreadNum;
    for(int iter=0;iter<num_iters;iter++){
        int i = iter * kThreadNum + tid;
        float val = i<block_nums?blocks[i]:0.0f;
        val = warp_prefix_sum(val);
        if(lane_id==31)
            smem[warp_id] = val;
        __syncthreads();
        if(warp_id==0){
            float tmp = lane_id<kThreadNum/kWarpSize?smem[lane_id]:0.0f;
            tmp = warp_prefix_sum(tmp);
            if(lane_id<kThreadNum/kWarpSize) smem[lane_id] = tmp;
        }
        __syncthreads();
        float bias = warp_id>0?smem[warp_id-1]:0.0f;
        if(i<block_nums) blocks[i] = val + bias + pre_acc;
        if(tid==0) pre_acc += smem[kThreadNum/kWarpSize-1];
        __syncthreads();
    }
}

__global__ void add_bias_kernel(float* output,float* blocks,int block_nums,int n){
    // [BUG FIX] 原始代码:
    //   __shared__ float block_bias;
    //   for(int block_id=blockIdx.x;block_id<block_nums;block_id+=gridDim.x){
    //       if(block_id==0) continue;
    //       if(threadIdx.x==0) block_bias = blocks[block_id-1];
    //       __syncthreads();
    //       for(int i=...) {
    //           output[base+j] += block_bias;
    //       }
    //       // 缺少 __syncthreads()!
    //   }
    //
    // 原因: 当 gridDim.x < block_nums 时, 一个 CUDA block 需要循环处理多个 data block。
    //       内层循环后没有 __syncthreads(), warp 0 (含 thread 0) 可能先跑完进入下一轮迭代,
    //       覆盖 block_bias, 而其他 warp 还在上一轮读旧的 block_bias。
    //
    // 修正: 移除 shared memory, 每个线程直接从 global memory 读 bias。
    //       blocks[block_id-1] 已在 L2 cache 中, 128 个线程读同一地址会被硬件合并,
    //       性能损失极小, 且彻底消除了 shared memory 竞态风险。
    //       如果后续确认需要 shared memory 优化, 必须在内层循环后加 __syncthreads()。
    for(int block_id=blockIdx.x;block_id<block_nums;block_id+=gridDim.x){
        if (block_id==0){
            continue;
        }
        float bias = blocks[block_id-1];
        for(int i=threadIdx.x; i<perBlockNum/kElemst; i+=kThreadNum){
            int base = block_id*perBlockNum+i*kElemst;
            if(base+kElemst<=n){
                float4 vec = *reinterpret_cast<float4*>(&output[base]);
                vec.x += bias;
                vec.y += bias;
                vec.z += bias;
                vec.w += bias;
                *reinterpret_cast<float4*>(&output[base]) = vec;
            }else{
                for(int j=0;j<kElemst;j++){
                    if(base+j<n) output[base+j] += bias;
                }
            }
            // for(int j=0;j<kElemst;j++){
            //     if(base+j<n) output[base+j] += bias;
            // }
        }
    }
}

torch::Tensor scan_then_fan(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto shape = input.sizes();
    auto flat = input.reshape(-1);
    size_t n = flat.size(0);
    if (n == 0) {
        return input.clone();
    }

    auto output = torch::empty_like(flat);
    dim3 block_dim;
    block_dim.x = kThreadNum;
    dim3 grid_dim;
    int block_nums = (n + perBlockNum - 1) / perBlockNum;
    grid_dim.x = min(block_nums,4096);
    auto blocks = torch::empty({(int64_t)block_nums}, flat.options());
    pre_block_prefix_sum<<<grid_dim,block_dim>>>(input.data_ptr<float>(),output.data_ptr<float>(),blocks.data_ptr<float>(),block_nums,n);
    prefix_sum_blocks<<<1,kThreadNum>>>(blocks.data_ptr<float>(),block_nums);
    add_bias_kernel<<<grid_dim,block_dim>>>(output.data_ptr<float>(),blocks.data_ptr<float>(),block_nums,n);
    return output;
}

