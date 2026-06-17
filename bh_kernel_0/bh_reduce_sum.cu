#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>
constexpr int kElemetsize = 4;
constexpr int kThreads = 128;
constexpr int kPerBlock = 1024;

__device__ float reduce_sum_warp(float val){
    for(int offset=16;offset>0;offset=offset/2){
        val+=__shfl_xor_sync(0xFFFFFFFF,val,offset);
    }
    return val;
}
__global__ void reduce_sum_kernel(
    float* input,
    float* output,
    int N,
    int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int col_id = blockIdx.y;
    int lane_id = tid%32;
    int warp_id = tid/32;
    int num_warps = kThreads/32;
    
    float val = 0.0f;
    int base_inedx = bid*D+col_id*kPerBlock;
    int chunk_star = col_id*kPerBlock;
    int chunk_end = min(col_id*kPerBlock+kPerBlock,D);
    int chunk_size = chunk_end - chunk_star;
    int VEC_SIZE = chunk_size/kElemetsize;
    
    for(int i=tid;i<VEC_SIZE;i+=kThreads){
        for(int j=0;j<kElemetsize;j++){
            val+=input[base_inedx+i*kElemetsize+j];
        }
    }
    //单元素处理，处理剩余的元素
    int tail_start = VEC_SIZE * kElemetsize;                                                                                                                                                                                      
    int tail_count = chunk_size - tail_start;
    if(tid<tail_count){
        val += input[base_inedx + tail_start + tid]; 
    }

    val=reduce_sum_warp(val);
    __shared__ float smem[32];
    if(lane_id==0)
        smem[warp_id] = val;
    __syncthreads();
    if(warp_id==0){
        float tmp_val = lane_id<num_warps?smem[lane_id]:0.0f;
        tmp_val = reduce_sum_warp(tmp_val);
        if(lane_id==0)
            atomicAdd(&output[bid],tmp_val);
    }


}
__global__ void reduce_sum_kernel_new(float *input,float* output,int N,int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid%32;
    int warp_id = tid/32;
    constexpr int num_warps = 128/32;
    int num_iter = (D+kThreads*kElemetsize -1)/(kThreads*kElemetsize);
    __shared__ float smem[num_warps];
    __shared__ float res;
    if(tid==0) res=0.0f;
    __syncthreads();
    for(int iter = 0; iter<num_iter;iter++){
        float sum=0.0f;
        int base_inedx = bid*D;
        int d_offset = iter*kThreads*kElemetsize+tid*kElemetsize;
        if(d_offset+kElemetsize<=D && ((base_inedx+d_offset)%kElemetsize==0)){
            float4 tmp = *reinterpret_cast<float4*>(&input[base_inedx+d_offset]);
            sum = tmp.x + tmp.y + tmp.z + tmp.w;
        }else{
            for(int i=0;i<kElemetsize;i++){
                if(d_offset+i<D) sum += input[base_inedx+d_offset+i];
            }
        }
        sum = reduce_sum_warp(sum);
        if(lane_id==0){
            smem[warp_id]=sum;
        }
        __syncthreads();
        if(warp_id==0){
            float tmp = lane_id<num_warps?smem[lane_id]:0.0f;
            tmp = reduce_sum_warp(tmp);
            if(lane_id<num_warps) smem[lane_id]=tmp;
        }
        __syncthreads();
        if(tid == 0) res += smem[0];
        __syncthreads();
    }
    if(tid==0) output[bid]=res;
}
torch::Tensor reduce_sum(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    // const int kPackSize = 16/sizeof(float);

    auto output = torch::zeros({N}, input.options());
    int threads = 128;
    dim3 grid_dims;
    // grid_dims.x = N;
    // grid_dims.y = (D + kPerBlock -1)/ kPerBlock;
    // reduce_sum_kernel<<<grid_dims, threads>>>(
        // input.data_ptr<float>(),
        // output.data_ptr<float>(),
        // N, D
    // );
    grid_dims.x = N;
    reduce_sum_kernel_new<<<grid_dims,threads>>>(input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D);
    return output;
}

