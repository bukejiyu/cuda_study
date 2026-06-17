  #include <torch/extension.h>                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                            
  #define TILE_DIM 32                                                                                                                                                                                                                                                                                                       
  #define BLOCK_ROWS 8                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                            
  // 每个 block 处理 32x32 的 tile                                                                                                                                                                                                                                                                                          
  // 32x8 个线程，每个线程负责 tile 中 4 个元素（列方向跨步）                                                                                                                                                                                                                                                               
  __global__ void transpose_kernel(const float* input, float* output, int N, int D) {                                                                                                                                                                                                                                       
      // +1 消除 shared memory bank conflicts                                                                                                                                                                                                                                                                               
      __shared__ float tile[TILE_DIM][TILE_DIM + 1];                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                            
      // --- Phase 1: 从 input 合并读入 shared memory ---                                                                                                                                                                                                                                                                   
      int x = blockIdx.x * TILE_DIM + threadIdx.x;  // 列方向  D                                                                                                                                                                                                                                                         
      int y = blockIdx.y * TILE_DIM + threadIdx.y;  // 行方向  N                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                            
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {                                                                                                                                                                                                                                                                      
          if (x < D && y + j < N)                            
            //input[(y + j) * D + x] 因为一个warp 32个线程 x就是线程方向，这时候读的值全部都是连续的，kernel中 thread 按x->y->z 的维度在改变          
            //threadIdx.x 是最快变化的维度    
            //一个 warp 是 tid 0~31，在 (32, 8) 的 block 里：
            //tid 0~31:                                                                                                                                                                                                                                                                                                                 
            //threadIdx.x = 0, 1, 2, ..., 31                                                                                                                                                                                                                                                                                          
            //threadIdx.y = 0, 0, 0, ..., 0                                                                                                                                                                                                                                               
              tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * D + x];                                                                                                                                                                                                                                                  
      }                                                                                                                                                                                                                                                                                                                     
      __syncthreads();                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                            
      // --- Phase 2: 从 shared memory 转置写出，合并写 ---                                                                                                                                                                                                                                                                 
      // 关键：blockIdx.x/y 互换，对应转置位置                                                                                                                                                                                                                                                                              
      x = blockIdx.y * TILE_DIM + threadIdx.x; // 是 N 输出的列                                                                                                                                                                                                                                                        
      y = blockIdx.x * TILE_DIM + threadIdx.y; // 是 D  输出的行                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                            
      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {                                                                                                                                                                                                                                                                      
          if (x < N && y + j < D)                                                                                                                                                                                                                                                                                           
              output[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];  
            //bank    (tid.x *33+tid.y+j)%32  ,因为x优先 这时候 tid.y+j是常量   
            //如果是32*32 那bank地址就变成了       
            // (tid.x *32+tid.y+j)%32      
            // 这时候 变成(tid.y+j)%32,当按x进行事务合并的时候 这是个常量，所有的线程都会访问相同的bank出现冲突                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                                                     
  }                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                            
  torch::Tensor transpose(torch::Tensor input) {                                                                                                                                                                                                                                                                            
      TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
      TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                            
      int N = input.size(0);                                                                                                                                                                                                                                                                                                
      int D = input.size(1);                                                                                                                                                                                                                                                                                                
      auto output = torch::zeros({D, N}, input.options());                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                            
      dim3 block(TILE_DIM, BLOCK_ROWS);                                                                                                                                                                                                                                                                                     
      dim3 grid((D + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                            
      transpose_kernel<<<grid, block>>>(                                                                                                                                                                                                                                                                                    
          input.data_ptr<float>(),                                                                                                                                                                                                                                                                                          
          output.data_ptr<float>(),                                                                                                                                                                                                                                                                                         
          N, D                                                                                                                                                                                                                                                                                                              
      );                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                            
      return output;                                                                                                                                                                                                                                                                                                        
  }     





__global__ void transpose_kernel(float* input float* output,int N,int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid/32;
    int lane_id = tid%32;

    for (int i=tid;i<N;i += blockDim.x){
        int dst_id = bid*N+i;
        int src_id = i*D + bid;
        output[dst_id] = input[src_id];
    }
}

#define TILE_DIM 32
#define BLOCK_ROWS 8 

__global__ void transpose_smem_kernel(float* input float* output,int N,int D){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid/32;
    int lane_id = tid%32;

    for (int i=tid;i<N;i += blockDim.x){
        int dst_id = bid*N+i;
        int src_id = i*D + bid;
        output[dst_id] = input[src_id];
    }
}

torch::Tensor transpose(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");

    int N = input.size(0);
    int D = input.size(1);
    // const int kPackSize = 16/sizeof(float);

    auto output = torch::zeros({D,N}, input.options());
    int threads = 128;
    dim3 grid_dims;
    grid_dims.x = D;

    // int elements_per_block = 1024;
    // int blocks_per_row = (D + elements_per_block - 1) / elements_per_block;



    // K_nums /  kPackSize
    // constexpr int tokens_per_block = K_nums /  kPackSize;

    // dim3 grid_dims;
    // grid_dims.x = D;

    transpose_kernel<<<grid_dims, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,D
    );

    return output;
}