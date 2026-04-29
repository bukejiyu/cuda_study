#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>

__inline__ __device__ float warp_reduc(float val){
    for (int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_sum_kernel(const float* input, float* output, int N, int D){
    const int VEC_SIZE = 16 / sizeof(float);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    float val = 0.0;

    const float* row_ptr = input + bid * D;
    
    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
    int D_vec = D / VEC_SIZE;

    for(int i=tid;i<D_vec;i+=blockDim.x){
        float4 v = row_ptr4[i];
        val+= v.x + v.y + v.z + v.w;
    }

    for (int i= D_vec*VEC_SIZE+tid;i<D;i+=blockDim.x){
        val += row_ptr[i];
    }
    

    val = warp_reduc(val);

    __shared__ float warp_vals[32];

    if(lane == 0){
        warp_vals[warp_id] = val;
    }
    __syncthreads();

    if(warp_id == 0){
        val = lane < num_warps ? warp_vals[lane] : 0.0;
        val = warp_reduc(val);
        if (lane == 0){
            output[bid] = val;
        }
    }
}

torch::Tensor reduce_sum(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    // const int KTypeSize = 16/sizeof(float);

    auto output = torch::zeros({N}, input.options());

    int threads = 256;

    int blocks = N;

    reduce_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce sum along last dim (CUDA)");
}
