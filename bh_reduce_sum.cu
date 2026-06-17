#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>

__inline__ __device__ float warp_reduc(float val){
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//这块专门做 block归约 
__global__ void reduce_sum_kernel(const float* input, float* output, int N, int D, int blocks_per_row){
    const int VEC_SIZE = 16 / sizeof(float);
    int bid = blockIdx.x;
    int oid = blockIdx.y;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    float val = 0.0f;
    int elements_per_block = 1024;

    int start = oid * elements_per_block;
    int end = min(start + elements_per_block, D);
    int chunk_size = end - start;

    const float* row_ptr = input + bid * D + start;

    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
    int D_vec = chunk_size / VEC_SIZE;

    for(int i=tid;i<D_vec;i+=blockDim.x){
        float4 v = row_ptr4[i];
        val+= v.x + v.y + v.z + v.w;
    }

    for (int i= D_vec*VEC_SIZE+tid;i<chunk_size;i+=blockDim.x){
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
            atomicAdd(&output[bid], val);
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
    int elements_per_block = 1024;
    int blocks_per_row = (D + elements_per_block - 1) / elements_per_block;



    // K_nums /  kPackSize
    // constexpr int tokens_per_block = K_nums /  kPackSize;

    dim3 grid_dims;
    grid_dims.x = N;
    grid_dims.y = blocks_per_row;

    reduce_sum_kernel<<<grid_dims, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D, blocks_per_row
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce sum along last dim (CUDA)");
}
