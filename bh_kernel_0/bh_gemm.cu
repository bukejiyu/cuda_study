#include <torch/extension.h>
constexpr int tile_m = 32;
constexpr int tile_n = 32;
constexpr int tile_k = 32;

__global__ void navie_gemm_fp32(float* A, float* B,float* C,int M,int N,int K){
    int c_row = blockIdx.y*blockDim.y+threadIdx.y;
    int c_col = blockIdx.x*blockDim.x+threadIdx.x;
    float sum = 0.0f;
    if(c_row < M && c_col < N){
        for(int k = 0 ; k < K ; k++){
            sum += A[c_row * K + k] * B[k * N + c_col];
        }
        C[c_row*N+c_col] = sum;
    }
}
torch::Tensor navie_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(A.dim() == 2, "MK");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::zeros({M,N}, A.options());
    dim3 block(tile_n,tile_m);
    dim3 grid((N + tile_n - 1)/tile_n,(M + tile_m - 1)/tile_m);
    navie_gemm_fp32<<<grid,block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );
    return C;
}


constexpr int PAD = 4;

__global__ void smem_gemm_fp32(float* A, float* B,float* C,int M,int N,int K){
    __shared__ float a_smem[tile_m][tile_k + PAD];
    __shared__ float b_smem[tile_k][tile_n + PAD];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int seme_a_row = tid/tile_m;
    int seme_a_col = tid%tile_k;
    int seme_b_row = tid/tile_k;
    int seme_b_col = tid%tile_n;
    int c_row = blockDim.y*blockIdx.y;
    int c_col = blockDim.x*blockIdx.x;
    float sum = 0.0f;

    for(int k = 0; k < K ; k += tile_k){
        int a_tile_col = k + seme_a_col;
        int a_tile_row = c_row + seme_a_row;
        a_smem[seme_a_row][seme_a_col] = a_tile_row<M && a_tile_col<K ? A[a_tile_row * K + a_tile_col] : 0.0f;
        int b_tile_row =  k + seme_b_row ;
        int b_tile_col = c_col + seme_b_col;
        b_smem[seme_b_row][seme_b_col] = b_tile_row<K && b_tile_col<N ? B[b_tile_row*N+b_tile_col] : 0.0f;
        __syncthreads();
#pragma unroll
        for(int ki=0;ki<tile_k;ki++){
            float a = a_smem[threadIdx.y][ki];
            float b = b_smem[ki][threadIdx.x];
            sum += a * b;
        }
        __syncthreads();
    }
    int row = c_row+threadIdx.y;
    int col = c_col+threadIdx.x;
    if(row < M && col < N) C[row*N+col]=sum;
}
torch::Tensor smem_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(A.dim() == 2, "MK");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::zeros({M,N}, A.options());
    dim3 block(tile_n,tile_m);
    dim3 grid((N + tile_n - 1)/tile_n,(M + tile_m - 1)/tile_m);
    smem_gemm_fp32<<<grid,block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );
    return C;
}

//整个思路为了增加每个线程的计算量，以前的每个线程 取2个float 1个乘1个加 结束了
//A[M,K]
//B[K,N]
// 增大的想法就是 让每个线程多处理一些 数据 这个case的目的就是想让每个线程处理8个份（row/col）
//因此a_smem 也很自然的 另外每个线程 处理8份数据 那么频繁的L1/Cache的访问 就太多了
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void smem_t8x8_gemm_fp32x4(float* A, float* B,float* C,int M,int N,int K){
    int c_row = blockIdx.y*BM;//这是c的当前block处理的row
    int c_col = blockIdx.x*BN;//这是c的当前block处理的col
    //128,8
    //另外读和写 我都是 float4
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];
    //tid [0-255]
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_a_row = tid/2;
    int smem_a_col = (tid%2)*4;
    //128/4=32
    int smem_b_row = tid/32;
    int smem_b_col = (tid%32)*4;

    int k_tile = (K+BK-1)/BK;
    float r_mn[TM][TN]={0.0f};
    for(int k=0;k<k_tile;k++){
        int a_row = c_row + smem_a_row;
        int a_col = k*BK + smem_a_col;
        // 加载 A 到 smem，处理边界和对齐
        if(a_row < M && a_col < K){
            int remain = K - a_col;
            // 检查 16B 对齐：((a_row*K + a_col)*4) % 16 == 0
            // 即 (a_row*K + a_col) % 4 == 0，a_col 是 4 的倍数，所以检查 a_row*K
            bool aligned = ((a_row * K) % 4) == 0;
            if(remain >= 4 && aligned){
                float4 val = *reinterpret_cast<float4*>(&A[a_row*K+a_col]);
                a_smem[smem_a_row][smem_a_col]=val.x;
                a_smem[smem_a_row][smem_a_col+1]=val.y;
                a_smem[smem_a_row][smem_a_col+2]=val.z;
                a_smem[smem_a_row][smem_a_col+3]=val.w;
            }else{
                for(int i=0;i<4;i++){
                    a_smem[smem_a_row][smem_a_col+i] = (a_col+i<K) ? A[a_row*K+a_col+i] : 0.0f;
                }
            }
        }else{
            a_smem[smem_a_row][smem_a_col]=0.0f;
            a_smem[smem_a_row][smem_a_col+1]=0.0f;
            a_smem[smem_a_row][smem_a_col+2]=0.0f;
            a_smem[smem_a_row][smem_a_col+3]=0.0f;
        }
        //bank 分析  每个线程 要写4个bank (8*m+k)%32 -> ((tid/2) * 8 + (tid%2)*4) % 32
        //y=0 x=0 tid =0 bank = 0
        //y=0 x=1 tid =1  bank = 4
        //y=0 x=2. tid=8  bank = 8 所以可以知道 这是4路写冲突
        int b_row = k * BK + smem_b_row;
        int b_col = c_col + smem_b_col;
        // 加载 B 到 smem，处理边界和对齐
        if(b_row < K && b_col < N){
            int remain = N - b_col;
            // 检查 16B 对齐：((b_row*N + b_col)*4) % 16 == 0
            // 即 (b_row*N + b_col) % 4 == 0，b_col 是 4 的倍数，所以检查 b_row*N
            bool aligned = ((b_row * N) % 4) == 0;
            if(remain >= 4 && aligned){
                float4 val_b = *reinterpret_cast<float4*>(&B[b_row*N+b_col]);
                b_smem[smem_b_row][smem_b_col] = val_b.x;
                b_smem[smem_b_row][smem_b_col + 1] = val_b.y;
                b_smem[smem_b_row][smem_b_col + 2] = val_b.z;
                b_smem[smem_b_row][smem_b_col + 3] = val_b.w;
            }else{
                for(int i=0;i<4;i++){
                    b_smem[smem_b_row][smem_b_col+i] = (b_col+i<N) ? B[b_row*N+b_col+i] : 0.0f;
                }
            }
        }else{
            b_smem[smem_b_row][smem_b_col] = 0.0f;
            b_smem[smem_b_row][smem_b_col + 1] = 0.0f;
            b_smem[smem_b_row][smem_b_col + 2] = 0.0f;
            b_smem[smem_b_row][smem_b_col + 3] = 0.0f;
        }
        //bank 分析 (128*row+col)%32 -> (tid/32*128+(tid%32)*4)%32
        //y=0 x=0 tid=0 bank=0,1,2,3
        //y=0,x=1 tid=1 bank=4,5,6,7 一样的4路写冲突
        __syncthreads();
        //上面是所有线程 搬运 128*K * 2 大小的数据到smem

        //下面是每个线程 分别处理8*8 这时候就是在 分割 128*128的矩阵
        //怪不得这个操作叫 tile_thread,线程的计算量变大了
#pragma unroll
        for(int i=0;i<BK;i++){
#pragma unroll
            for(int m=0;m<TM;m++){
#pragma unroll
                for(int n=0;n<TN;n++){
                    int tmp_row = m + threadIdx.y*TM;
                    int tmp_col = n + threadIdx.x*TN;
                    if(tmp_row<M && tmp_col<N){
                        r_mn[m][n] += a_smem[tmp_row][i] * b_smem[i][tmp_col];
                    }
                }
            }
        }
        __syncthreads();
    }
    for(int m = 0;m < TM;m++){
        int c_row_now =  c_row + threadIdx.y * TM + m;
        if(c_row_now < M){
            for(int n=0;n<TN;n+=4){
                int c_col_now = c_col + threadIdx.x*TN + n;
                int remain = N - c_col_now;
                // 检查 16B 对齐：((c_row_now*N + c_col_now)*4) % 16 == 0
                // 即 (c_row_now*N + c_col_now) % 4 == 0，c_col_now 是 4 的倍数
                bool aligned = ((c_row_now * N) % 4) == 0;
                if(remain >= 4 && aligned){
                    float4 c_res = *reinterpret_cast<float4*>(&r_mn[m][n]);
                    *reinterpret_cast<float4*>(&C[c_row_now*N+c_col_now]) = c_res;
                }else{
                    for(int i=0;i<remain;i++){
                        C[c_row_now*N+c_col_now+i] = r_mn[m][n+i];
                    }
                }
            }
        }
    }
}
torch::Tensor smem_t8x8_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(A.dim() == 2, "MK");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    //整个这个kernel的优化目的就是 增大计算量 每个线程的计算量
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    auto C = torch::zeros({M,N}, A.options());
    dim3 block(BN/TN,BM/TM);
    dim3 grid((N + BN - 1)/BN,(M + BM - 1)/BM);
    smem_t8x8_gemm_fp32x4<BM,BN,BK,TM,TN><<<grid,block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );
    return C;
}