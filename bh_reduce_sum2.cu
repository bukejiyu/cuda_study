#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>

inline __device__ float warp_sum(float val){
    #pragma unroll
    for(int offset=16;offset>0;offset=offset/2){
        val += __shfl_down_sync(0XFFFFFFFF,val,offset);
    }
    return val;
}

__global__ void reduce_sum_kernel(float *input,float *output,int N,int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id =  tid/32;
    int lane_id = tid%32;

    float res = 0.0f;
    float *input_data = input + bid*D;
    for(int i = tid ;i<D;i += blockDim.x){
        res += input_data[i];
    }
    res = warp_sum(res);

    __shared__ float smem[32];

    if(lane_id == 0){
        smem[warp_id] =  res;
    }

    __syncthreads();

    int num_warps = blockDim.x/32;
    if(warp_id == 0){
        float tmp_val = lane_id<num_warps?smem[lane_id]:0.0f;
        tmp_val=warp_sum(tmp_val);
        if(lane_id == 0){
            output[bid] = tmp_val;
        }
    }
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
    grid_dims.x = N;
    // int elements_per_block = 1024;
    // int blocks_per_row = (D + elements_per_block - 1) / elements_per_block;



    // K_nums /  kPackSize
    // constexpr int tokens_per_block = K_nums /  kPackSize;

    // dim3 grid_dims;
    // grid_dims.x = N;
    // grid_dims.y = blocks_per_row;

    reduce_sum_kernel<<<grid_dims, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    return output;
}