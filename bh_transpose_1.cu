 #include <torch/extension.h>                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                            
#define TILE_DIM 32                                                                                                                                                                                                                                                                                                       
#define BLOCK_ROWS 8 


__global__ void transpose_kernel(float* input ,float* output,int N,int D){
    int tid_row = threadIdx.x;
    int tid_col = threadIdx.y;

    int A_mat_row = blockIdx.y*TILE_DIM; 
    int A_mat_col = blockIdx.x*TILE_DIM;
    __shared__ float smem[TILE_DIM][TILE_DIM+1]; //设计成32 是因为 每个block 的线程数也只有 32*32 
    
    for(int i=0;i<TILE_DIM/BLOCK_ROWS;i++){
        if((A_mat_row + threadIdx.y*(TILE_DIM/BLOCK_ROWS) + i)<N && A_mat_col + threadIdx.x<D)
            smem[threadIdx.y*(TILE_DIM/BLOCK_ROWS) + i][threadIdx.x] = input[(A_mat_row + threadIdx.y*(TILE_DIM/BLOCK_ROWS) + i)*D +  A_mat_col + threadIdx.x];
        //读 input 
        //bank id = [(y*BLOCK_ROWS + i)*32+threadIdx.x]%32 -> threadIdx.x 写入没有bank冲突
    }
    __syncthreads();

    //这一步 做transpose 

    int B_mat_row = blockIdx.x*TILE_DIM;
    int B_mat_col = blockIdx.y*TILE_DIM;

    //为了写的时候合并事务 thread x应该与写的方向一致

    for(int i=0;i<TILE_DIM/BLOCK_ROWS;i++){
        if((B_mat_row+threadIdx.y*(TILE_DIM/BLOCK_ROWS) +i)<D && B_mat_col+threadIdx.x<N)
            output[(B_mat_row+threadIdx.y*(TILE_DIM/BLOCK_ROWS) +i)*N + B_mat_col+threadIdx.x] = smem[threadIdx.x][threadIdx.y*(TILE_DIM/BLOCK_ROWS) +i];
        //bank id = [threadIdx.x*32 + (threadIdx.y*BLOCK_ROWS +i)]%32 -> （0+c）导致bank冲突
    }

    //写 output 



    
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