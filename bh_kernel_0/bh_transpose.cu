  #include <torch/extension.h>                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                            
  #define TILE_DIM 32                                                                                                                                                                                                                                                                                                       
  #define BLOCK_ROWS 8   
  constexpr int kElemt = 4;


__global__ void transpose_kernel(float* input,float* output,int N,int D){
    int a_row_offset = blockIdx.x * TILE_DIM + threadIdx.y;
    int a_col_offset = blockIdx.y * TILE_DIM + threadIdx.x;

    __shared__ float smem[TILE_DIM][TILE_DIM+1];
#pragma unroll
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
#pragma unroll    
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

__global__ void transpose_v2_kernel(float* input,float* output,int N,int D){
    int a_row_offset = blockIdx.x * TILE_DIM ;
    int a_col_offset = blockIdx.y * TILE_DIM * kElemt ;

    __shared__ float smem[TILE_DIM][TILE_DIM*kElemt];
#pragma unroll  
    for(int i=0;i<TILE_DIM;i+=BLOCK_ROWS){
        int a_row_thread_offset = threadIdx.y + i;
        int a_col_thread_offset = threadIdx.x * kElemt;
        if((a_col_offset + a_col_thread_offset) < D && (a_row_offset + a_row_thread_offset) < N){
            if(a_col_offset + a_col_thread_offset + kElemt <= D && (((a_row_offset + a_row_thread_offset) * D)%4)==0 ){
                float4 tmp_data =*reinterpret_cast<float4*>(&input[(a_row_offset + a_row_thread_offset)*D+a_col_offset + a_col_thread_offset]);
                smem[threadIdx.y+i][threadIdx.x*kElemt] = tmp_data.x;
                smem[threadIdx.y+i][threadIdx.x*kElemt+1] = tmp_data.y;
                smem[threadIdx.y+i][threadIdx.x*kElemt+2] = tmp_data.z;
                smem[threadIdx.y+i][threadIdx.x*kElemt+3] = tmp_data.w;
            }else{
                for(int j=0 ; j < kElemt ; j++){
                    if((a_col_offset + a_col_thread_offset + j)<D&&(a_row_offset + a_row_thread_offset)<N) smem[threadIdx.y+i][threadIdx.x*kElemt+j]=input[(a_row_offset + a_row_thread_offset) * D+ a_col_offset + a_col_thread_offset + j];
                }
            }
        }
    }
    __syncthreads();

    constexpr int perElemt = TILE_DIM / kElemt;  // 8
    int krow = blockDim.x / perElemt;            // 4
    int b_row_offset = blockIdx.y * TILE_DIM * kElemt;
    int b_col_offset = blockIdx.x * TILE_DIM;

#pragma unroll
    for(int i = 0; i < TILE_DIM * kElemt; i += BLOCK_ROWS * krow) {
        int local_row = i + threadIdx.y * krow + threadIdx.x / perElemt;
        int local_col = (threadIdx.x % perElemt) * kElemt;

        int g_row = b_row_offset + local_row;
        int g_col = b_col_offset + local_col;

        if(g_row < D && g_col < N) {
            if(g_col + kElemt <= N && (g_row * N) % 4 == 0) {
                float4 tmp;
                tmp.x = smem[local_col + 0][local_row];
                tmp.y = smem[local_col + 1][local_row];
                tmp.z = smem[local_col + 2][local_row];
                tmp.w = smem[local_col + 3][local_row];
                *reinterpret_cast<float4*>(&output[g_row * N + g_col]) = tmp;
            } else {
                for(int j = 0; j < kElemt; j++) {
                    if(g_col + j < N)
                        output[g_row * N + g_col + j] = smem[local_col + j][local_row];
                }
            }
        }
}

}

torch::Tensor transpose_v2(torch::Tensor input) {
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
    grid_dims.y = (D + TILE_DIM*kElemt - 1) / (TILE_DIM*kElemt);

    transpose_v2_kernel<<<grid_dims, block_dims>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,D);

    return output;
}

__global__ void transpose_v3_kernel(float* input,float* output,int N,int D){
    float vec[4][kElemt];
    // BUG1: blockDim -> blockIdx
    // int a_row_offset = blockDim.x * TILE_DIM ;
    // int a_col_offset = blockDim.y * TILE_DIM * kElemt ;
    int a_row_offset = blockIdx.x * TILE_DIM ;
    int a_col_offset = blockIdx.y * TILE_DIM * kElemt ;

    for(int i=0;i<TILE_DIM/BLOCK_ROWS;i++){
        int row_thread_bias = threadIdx.y * TILE_DIM/BLOCK_ROWS + i;
        int col_thread_bias = threadIdx.x * kElemt;
        int a_row = a_row_offset + row_thread_bias;
        int a_col = a_col_offset + col_thread_bias;
        if(a_row<N&&a_col<D){
            if(a_col+kElemt<=D&&(a_row*D)%4==0){
                float4 tmp = *reinterpret_cast<float4*>(&input[a_row*D+a_col]);
                vec[i][0] = tmp.x;
                vec[i][1] = tmp.y;
                vec[i][2] = tmp.z;
                vec[i][3] = tmp.w;
            }else{
                for(int j=0;j<kElemt;j++){
                    // BUG2: a_col*D+a_row+j -> a_row*D+a_col+j (row/col反了)
                    // float tmp = a_col+j<D&&a_row<N?input[a_col*D+a_row+j]:0.0f;
                    float tmp = a_col+j<D&&a_row<N?input[a_row*D+a_col+j]:0.0f;
                    vec[i][j] = tmp;
                }
            }
        }else{
            for(int j=0;j<kElemt;j++) vec[i][j] = 0.0f;
        }
    }

    // BUG3: blockDim -> blockIdx
    // int b_row_offset = blockDim.y * TILE_DIM * kElemt ;
    // int b_col_offset = blockDim.x * TILE_DIM ;
    int b_row_offset = blockIdx.y * TILE_DIM * kElemt ;
    int b_col_offset = blockIdx.x * TILE_DIM ;

    // 原始写法: row覆盖不够，float4打包错误
    // for(int i=0;i<TILE_DIM/BLOCK_ROWS;i++){
    //     int row_thread_bias = threadIdx.x * kElemt;
    //     int col_thread_bias = threadIdx.y * TILE_DIM/BLOCK_ROWS + i;
    //     int b_row = b_row_offset + row_thread_bias;
    //     int b_col = b_col_offset + col_thread_bias;
    //     if(b_row < D && b_col < N){
    //         if(b_col+kElemt<N&&(b_row*N+b_col)%16==0){
    //             float4 tmp;
    //             tmp.x = vec[threadIdx.x%kElemt][i];
    //             tmp.y = vec[threadIdx.x%kElemt][i];
    //             tmp.z = vec[threadIdx.x%kElemt][i];
    //             tmp.w = vec[threadIdx.x%kElemt][i];
    //             *reinterpret_cast<float4*>(&output[b_row*N+b_col]) = tmp;
    //         }else{
    //             for(int j=0;j<kElemt;j++){
    //                 float tmp = b_col+j<N?vec[j][i]:0.0f;
    //                 output[b_row*N+b_col+j] = tmp;
    //             }
    //         }
    //     }
    // }

    // 修正: tx管行(转置后列变行), ty管列
    // vec[i][k] = input[row+i, col+k] → output[col+k, row+i] = vec[i][k]
    // b_row = tx*4+k (k循环展开), b_col = ty*4+i, float4沿k方向取vec[i][k]
    for(int k=0; k<kElemt; k++){
        int b_row = b_row_offset + threadIdx.x * kElemt + k;
        int b_col = b_col_offset + threadIdx.y * kElemt;
        if(b_row < D && b_col < N){
            if(b_col+kElemt<=N && (b_row*N)%4==0){
                float4 tmp;
                tmp.x = vec[0][k];
                tmp.y = vec[1][k];
                tmp.z = vec[2][k];
                tmp.w = vec[3][k];
                *reinterpret_cast<float4*>(&output[b_row*N+b_col]) = tmp;
            }else{
                for(int j=0;j<kElemt;j++){
                    if(b_col+j<N)
                        output[b_row*N+b_col+j] = vec[j][k];
                }
            }
        }
    }

}
torch::Tensor transpose_v3(torch::Tensor input) {
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
    grid_dims.y = (D + TILE_DIM*kElemt - 1) / (TILE_DIM*kElemt);

    transpose_v3_kernel<<<grid_dims, block_dims>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,D);

    return output;
}