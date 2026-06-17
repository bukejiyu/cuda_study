#include <torch/extension.h>                                                                                                                                                                                                                                                                                              
#define TILE_DIM 32
#define BLOCK_ROWS 8


__global__ void transpose_kernel(float* input, float* output,int N,int D){

    int a_row = blockIdx.y*TILE_DIM +threadIdx.y;
    int a_col = blockIdx.x*TILE_DIM +threadIdx.x;

    __shared__ float smem[TILE_DIM][TILE_DIM+1];
    //gpu的并行思想是每个线程干一份活，如果干完了就去干下一个区间的活，而不是 一个线程干完一个区间的活
    for(int j = 0;j < TILE_DIM;j+=BLOCK_ROWS){
        if(a_row+j<N&&a_col<D)
            smem[threadIdx.y+j][threadIdx.x]=input[(a_row+j)*D+a_col];
    }
    __syncthreads();  
    int b_row = blockIdx.x*TILE_DIM +  threadIdx.y;
    int b_col = blockIdx.y*TILE_DIM +  threadIdx.x;

    for(int j=0;j<TILE_DIM;j+=BLOCK_ROWS){
        if(b_row+j<D&&b_col<N)
            output[(b_row+j)*N+b_col]=smem[threadIdx.x][threadIdx.y+j];
    }
}

torch::Tensor transpose(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    // const int kPackSize = 16/sizeof(float);

    auto output = torch::zeros({D,N}, input.options());
    dim3 block_dims;
    dim3 grid_dims;

    block_dims.x = TILE_DIM;
    block_dims.y = BLOCK_ROWS;

    grid_dims.x = (D + TILE_DIM -1)/TILE_DIM ;
    grid_dims.y = (N + TILE_DIM -1)/TILE_DIM ;

    transpose_kernel<<<grid_dims, block_dims>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    return output;
}