#include <torch/extension.h>

constexpr int kThreads = 256;
constexpr int kElemts = 4;
__device__ float warp_reduce(float data){
    for(int offset = 16;offset>0;offset=offset/2){
        data += __shfl_xor_sync(0xFFFFFFFF,data,offset);
    }
    return data;
}
__global__ void group_norm_kernel_v2(const float* input, float* output, const float* gamma, const float* beta, int N, int C, int H, int W, int num_groups){
    int tid = threadIdx.x;
    int bid = blockIdx.x/num_groups;
    int gid = blockIdx.x%num_groups;

    int lane_id = tid%32;
    int warp_id = tid/32;
    constexpr int num_warps = kThreads/32;

    int G = C/num_groups;
    int base = bid*C*H*W + gid*G*H*W;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int total_size = G*H*W; // 
    __shared__ float smem[num_warps];
    __shared__ float smem_sq[num_warps];
    int num_iter = (total_size+kElemts-1)/kElemts;
    for(int i = tid ; i < num_iter ; i += kThreads){
        int bias = i * kElemts;
        if(bias + kElemts <= total_size && (base*4)%16 == 0){
            const float4 tmp = *reinterpret_cast<const float4*>(&input[base+bias]);
            sum += tmp.x+tmp.y+tmp.z+tmp.w;
            sum_sq += tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z + tmp.w*tmp.w;
        }else{
            for(int j = 0;j < kElemts;j++){
                if(bias+j<total_size){
                    float data = input[base+bias+j];
                    sum+=data;
                    sum_sq+=data*data;
                }  
            }
        }
    }

    sum = warp_reduce(sum);
    sum_sq = warp_reduce(sum_sq);

    if(lane_id==0){
        smem[warp_id] = sum;
        smem_sq[warp_id] = sum_sq;
    }
    __syncthreads();
    if(warp_id==0){
        float tmp_sum =  lane_id<num_warps?smem[lane_id]:0.0f;
        float tmp_sum_sq =  lane_id<num_warps?smem_sq[lane_id]:0.0f;
        tmp_sum = warp_reduce(tmp_sum);
        tmp_sum_sq = warp_reduce(tmp_sum_sq);
        if(lane_id<num_warps){
            smem[lane_id] = tmp_sum;
            smem_sq[lane_id] = tmp_sum_sq;
        }
    }
    __syncthreads();
    float g_sum = smem[0];
    float g_sum_sq = smem_sq[0];
    __syncthreads();
    float mean = g_sum/total_size;
    float var = g_sum_sq/total_size - mean*mean;
    float inv_std = rsqrtf(var + 1e-5f);
    for(int i = tid;i < total_size;i += kThreads){
        int c_pergroup_id = i/(H*W);
        int global_c_id = c_pergroup_id + gid*G;
        float x = input[base+i];
        float x_norm = (x-mean)*inv_std;
        if(gamma != nullptr && beta != nullptr){
            x_norm = gamma[global_c_id]*x_norm+beta[global_c_id];
        }
        output[base+i] = x_norm;
    }

}

torch::Tensor group_norm_v2(torch::Tensor input, int num_groups,
                         c10::optional<torch::Tensor> gamma_opt,
                         c10::optional<torch::Tensor> beta_opt) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (NCHW)");
    TORCH_CHECK(input.size(1) % num_groups == 0, "C must be divisible by num_groups");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    torch::Tensor output = torch::zeros_like(input);
    TORCH_CHECK(C % num_groups ==  0, "C 需要能均分成 num_groups 组");
    dim3 grid; 
    grid.x = N*num_groups;   
    const float *gamma_ptr = gamma_opt.has_value() ? gamma_opt->data_ptr<float>() : nullptr;
    const float *beta_ptr  = beta_opt.has_value()  ? beta_opt->data_ptr<float>()  : nullptr;                     
    group_norm_kernel_v2<<<grid, kThreads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma_ptr,
        beta_ptr,
        N,C,H,W,num_groups
    );

    return output;
}
