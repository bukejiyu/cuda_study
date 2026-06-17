#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>

__device__ float warp_reduce_sum(float x){
    for (int offset=16;offset>0;offset=offset/2){
        x + =__shfl_down_sync(0xFFFFFFFF,x,offset);
    }
    return x;
}

__global__ void reduce_sum_kernel(float* input,int N,int D,float*output){
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int lane_id = tid%32;
    int bid = blockIdx.x;
    int num_warps = blockDim.x / 32;
    float res=0.0f;
    float* input_data = input + bid*D;
    for (int i=tid;i<D;i += blockDim.x){
        res += input_data[i]; 
    }

    float warp_res = warp_reduce_sum(res);
    __shared__ float smem[32];
    if(lane_id==0){
        smem[warp_id] = warp_res;
    }
    __syncthreads();
    if(warp_id==0){
        float tmp_seme = lane_id<num_warps?smem[lane_id]:0.0f;
        float final_res = warp_reduce_sum(tmp);
        if(lane_id==0){
            output[bid] = res;
        }
    }
}

