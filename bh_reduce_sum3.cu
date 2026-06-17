#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>

__device__ float warp_reduce(float data){
    #pragma unroll
    for(int offset=16;offset>0;offset=offset/2){
        data += __shfl_xor_sync(0xFFFFFFFF,data,offset);
    }
    return data;
}
__global__ void reduce_sum_kernel(float* input,float* output,int N,int D ){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid %32;
    int warp_id = tid /32;
    int num_warps = blockDim.x/32;

    float val = 0.0f;
    for(int i=tid;i<D;i += blockDim.x){ 
        val += input[bid*D+i];
    }
    val=warp_reduce(val);

    __shared__ float smem[32];
    if(lane_id==0){
        smem[warp_id]=val;
    }
    __syncthreads();

    if(warp_id==0){
        val = lane_id < num_warps ? smem[lane_id] : 0.0f;
        val = warp_reduce(val);
        if(lane_id==0)
            output[bid]=val;
    }
}

torch::Tensor reduce_sum(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    // const int kPackSize = 16/sizeof(float);

    auto output = torch::zeros({N}, input.options());
    // int threads = 128;
    // int elements_per_block = 1024;
    // int blocks_per_row = (D + elements_per_block - 1) / elements_per_block;
    int threads = 128;
    dim3 grid_dims;
    grid_dims.x = N;



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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce sum along last dim (CUDA)");
}