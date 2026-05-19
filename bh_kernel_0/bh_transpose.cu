  #include <torch/extension.h>                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                            
  #define TILE_DIM 32                                                                                                                                                                                                                                                                                                       
  #define BLOCK_ROWS 8   


__global__ void transpose_kernel(float* input,float* output,int N,int D){
    int a_row_offset = blockIdx.x * TILE_DIM + threadIdx.y;
    int a_col_offset = blockIdx.y * TILE_DIM + threadIdx.x;

    __shared__ float smem[TILE_DIM][TILE_DIM+1];
    for(int i=0;i<TILE_DIM;i+=BLOCK_ROWS){
        int tmp_row = a_row_offset + i;
        int tmp_col = a_col_offset;
        //读input 并且 让读方向连续 能合并事务
        if(tmp_row<N && tmp_col<D){
           smem[i+threadIdx.y][threadIdx.x] = input[tmp_row*D+tmp_col];
           //banck id = [(i+threadIdx.y)*33+threadIdx.x]%32 -> (c+threadIdx.x)%32 乌冲突
           //tmp_col访问global memory 连续
        }
    }
    __syncthreads();

    int b_row_offset = blockIdx.y * TILE_DIM + threadIdx.y;
    int b_col_offset = blockIdx.x * TILE_DIM + threadIdx.x;
    for(int i=0;i<TILE_DIM;i=i+BLOCK_ROWS){
        int tmp_row = b_row_offset + i;
        int tmp_col = b_col_offset;
        if(tmp_row<D&&tmp_col<N){
            output[tmp_row*N+tmp_col] = smem[threadIdx.x][i+threadIdx.y];
            //bank = [(threadIdx.x*33)+i+threadIdx.y]%32 -> [c+(1+threadIdx.x)]%32 无冲突
        }
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
    block_dims.x = TILE_DIM;
    block_dims.y = BLOCK_ROWS;
    dim3 grid_dims;
    grid_dims.x = (N + TILE_DIM - 1) / TILE_DIM;
    grid_dims.y = (D + TILE_DIM - 1) / TILE_DIM;

    transpose_kernel<<<grid_dims, block_dims>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,D
    );

    return output;
}