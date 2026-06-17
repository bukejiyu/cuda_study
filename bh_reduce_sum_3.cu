  #include <torch/extension.h>                                                                                                                                                                                                                                                                                              

__device__ float reduce_sum_warp(float data){
    #pragma unroll
    for(int offset=16;offset>0;offset=offset/2){
        data += __shfl_xor_sync(0xFFFFFFFF,data,offset);
    }
    return data;
}

//基本写法
__global__ void reduce_kernel(float* input ,float* output,int N,int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid/32;
    int lane_id = tid%32;
    int num_warps = blockDim.x/32;

    float res=0.0f;
    int row = bid*D;

    for(int i=tid;i<D;i += blockDim.x){
        res += input[row+i];
    }

    res = reduce_sum_warp(res);

    __shared__ float smem[32];

    smem[warp_id] = res;
    __syncthreads();

    if(warp_id==0){
        float smem_res = lane_id < num_warps ? smem[lane_id]:0.0f;
        smem_res = reduce_sum_warp(smem_res);
        if(lane_id==0){
            output[bid] = smem_res;
        }
    }

}


//这写法 向量化加载 降低带宽压力
__global__ void reduce_kernel_pack4(float* input ,float* output,int N,int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid/32;
    int lane_id = tid%32;
    int num_warps = blockDim.x/32;

    float res=0.0f;
    float4* input_vec = reinterpret_cast<float4*>(input + bid*D);                                                                                                                                                             
    int D_vec = D/4;

    // for(int i=tid;i<D/4&&i*4<D;i=i+blockDim.x){
    //     float4 vec = float4(input[bid*D+i]);
    //     res = res + vec.x+vec.y+vec.z+vec.w;
    // }
    for(int i=tid; i<D_vec; i+=blockDim.x){                                                                                                                                                                                   
        float4 vec = input_vec[i];                                                                                                                                                                                            
        res += vec.x + vec.y + vec.z + vec.w;                                                                                                                                                                                 
    } 

    int tail_start = D_vec * 4;

    for(int i=tid+tail_start;i<D;i += blockDim.x){
        res += input[bid*D + i];
    }

    res = reduce_sum_warp(res);
    __shared__ float smem[32];

    smem[warp_id] = res;
    __syncthreads();

    if(warp_id==0){
        float smem_res = lane_id < num_warps ? smem[lane_id]:0.0f;
        smem_res = reduce_sum_warp(smem_res);
        if(lane_id==0){
            output[bid] = smem_res;
        }
    }
}

__global__ void reduce_kernel_new(float* input,float* output,int N,int D,int elements_per_block){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int chunk_id = blockIdx.y;
    int lane_id = tid%32;
    int warp_id = tid/32;
    constexpr int NUM_THREADS = 128;
    constexpr int num_warps = NUM_THREADS/32;

    int chunk_star = bid*D+chunk_id*elements_per_block;
    int chunk_end = min((chunk_id+1)*elements_per_block,D);

    int chunk_size = chunk_end-chunk_id*elements_per_block;

    float res = 0.0f;
    for(int i=tid;i<chunk_size;i+=NUM_THREADS){
        res += input[chunk_star+i];
    }

    res = reduce_sum_warp(res);
    __shared__ float smem[32];
    smem[warp_id] = res; //每个block一个 smem
    __syncthreads();

    if(warp_id==0){
        float smem_res = lane_id < num_warps ? smem[lane_id]:0.0f;
        smem_res = reduce_sum_warp(smem_res);
        if(lane_id ==0){
            atomicAdd(&output[bid],smem_res);
        }
    }
    
}
//这个写法 为了让sm 利用率更多 ，虽然在reduce_sum 上没有啥优势
torch::Tensor reduce_sum_new(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    // const int kPackSize = 16/sizeof(float);

    auto output = torch::zeros({N}, input.options());
    int threads = 128;

    dim3 grid_dims;
    int elements_per_block = 1024;
    grid_dims.x = N;
    grid_dims.y = (D + elements_per_block -1) / elements_per_block;//向上取整


    reduce_kernel_new<<<grid_dims, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,D,elements_per_block
    );

    return output;
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

    transpose_kernel<<<grid_dims, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,D
    );

    return output;
}

torch::Tensor reduce_sum_pack4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    // const int kPackSize = 16/sizeof(float);

    auto output = torch::zeros({N}, input.options());
    int threads = 128/4;
    dim3 grid_dims;
    grid_dims.x = N;

    transpose_kernel<<<grid_dims, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,D
    );

    return output;
}