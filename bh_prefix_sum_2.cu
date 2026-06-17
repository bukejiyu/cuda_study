
#include <torch/extension.h>

constexpr int KElementsPerThreads = 4;
constexpr int kThreadNum = 256;
constexpr int kBlockSize = kThreadNum * KElementsPerThreads; 

__device__ float warp_prefix_sum(float val){
    for(int offset=1;offset<32;offset=offset*2){
        float tmp = __shfl_up_sync(0xFFFFFFFF,val,offset);
        if (threadIdx.x % 32 >= offset){
            val +=tmp;
        }
    }
    return val;
}

__device__ void prefix_sum_block(float* block_smem){
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    constexpr int nums_warp = kThreadNum/32;
    float val = block_smem[tid];
    //32个线程为一个warp的做前缀和
    val = warp_prefix_sum(val);
    __shared__ float warp_smem[nums_warp];
    if(lane_id==31){
        warp_smem[warp_id] = val;
    }
    __syncthreads();

    if(warp_id==0){
        float warp_val = lane_id<nums_warp? warp_smem[lane_id]:0.0f;
        warp_val = warp_prefix_sum(warp_val);
        if(lane_id<nums_warp)
            warp_smem[lane_id] = warp_val;
    }
    __syncthreads();

    float offset = warp_id>0 ? warp_smem[warp_id-1]:0.0f;

    block_smem[tid] = val + offset;
    __syncthreads();
}
__global__ void LocalScanKernel(float* input,float* blocks,float * output,int n ,int block_nums){

    for(int block_id = blockIdx.x;block_id < block_nums;block_id += gridDim.x){
        int tid = threadIdx.x;
        float data_vec[KElementsPerThreads];
        float acc = 0.0f;

        int base =  block_id * kBlockSize;

        for(int i = 0;i < KElementsPerThreads;i++){
            int idx = base + tid*KElementsPerThreads + i;
            float val = idx < n ? input[idx] : 0.0f;
            acc = val + acc;
            data_vec[i] = acc;
        }

        __shared__ float block_smem[kThreadNum];
        block_smem[tid]=acc;
        __syncthreads();
        prefix_sum_block(block_smem);

        float offset = tid>0? block_smem[tid-1]:0.0f;
        for(int i = 0;i < KElementsPerThreads;i++){
            int idx = base + tid*KElementsPerThreads + i;
            if (idx < n ){
                output[idx] = data_vec[i] + offset;
            }
        }
        if(tid==0) blocks[block_id] = block_smem[kThreadNum-1];
    }

}


//这个kernel的目的是让所有block 的tensor 变成完整累加的
__global__ void BlocksSumKernel(float* blocks,int block_num){
    int tid = threadIdx.x;
    __shared__ float smem[kThreadNum];
    __shared__ float base ;
    if(tid ==0){
        base = 0.0f;
    }
    __syncthreads();
    for (int i = 0;i<block_num;i+=kThreadNum){
        int index = i + tid;
        smem[tid] = index<block_num?blocks[index]:0.0f;
        __syncthreads();
        prefix_sum_block(smem);
        if(index<block_num)
            blocks[index] = smem[tid] + base;
        __syncthreads();

        if(tid == kThreadNum-1){
            base = base + smem[tid];
        }
        __syncthreads();
    }
    
}

__global__ void AddBaseOffsetKernel(float* blocks,float* output,int n,int block_nums){
    for(int block_id = blockIdx.x;block_id < block_nums;block_id += gridDim.x){
        float offset = block_id>0?blocks[block_id-1]:0.0f;
        for (int j = 0; j < KElementsPerThreads; ++j) {
            int idx = block_id * kBlockSize + threadIdx.x * KElementsPerThreads + j;
            if (idx < n) {
                output[idx] += offset;
            }
        }
    }
}
// tensor shape =[N,D]
//output shape N*D
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


    dim3 grid_dims;
    int block_nums = (n + kBlockSize -1)/kBlockSize;
    grid_dims.x = min(block_nums,1024);

    auto output = torch::zeros({n}, input.options());
    auto blocks = torch::zeros({block_nums}, input.options());


    LocalScanKernel<<<grid_dims, kThreadNum>>>(
        flat.data_ptr<float>(), blocks.data_ptr<float>(),
        output.data_ptr<float>(), n, block_nums);


    BlocksSumKernel<<<1, kThreadNum>>>(
        blocks.data_ptr<float>(), block_nums);

    AddBaseOffsetKernel<<<grid_dims, kThreadNum>>>(
        blocks.data_ptr<float>(), output.data_ptr<float>(), n, block_nums);

    return output.reshape(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_then_fan", &scan_then_fan, "Scan-then-fan prefix sum");
}
