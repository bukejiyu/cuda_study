#include <torch/extension.h>
//qkv [bs,1,head+2kv_head,dim] -> [token_len,head+2kv_head,dim]
//cos_emb [max_seq_len,head_dim/2]
//sin_emb [max_seq_len,head_dim/2]
// positions [bs]
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kElemt = 4;
constexpr int kHalfElemt =2;

__global__ void decode_apply_rope_kernel(
    float* __restrict__ qkv, 
    const float* __restrict__ cos_emb,//[max_seq_len,head_dim/2]
    const float* __restrict__ sin_emb,//[max_seq_len,head_dim/2]
    const int* __restrict__ positions,//[bs]
    int num_head,
    int kv_num_head,
    int head_dim,
    int elem_nums){
        int global_warp_id = blockDim.y * blockIdx.x + threadIdx.y; //这是所有的block 可以覆盖的所有 token_len*(num_head+2*kv_num_head)的大小
        int all_warps = gridDim.x *  blockDim.y; // 
        int total_head = elem_nums/head_dim;//
        int token_offset = (num_head+2*kv_num_head)*head_dim;
        //gloabl_hi 是一个很大的head 我个人理解是 
        for(int gloabl_hi = global_warp_id;gloabl_hi<total_head;gloabl_hi+=all_warps){
            int global_index = gloabl_hi * head_dim + threadIdx.x * kElemt;
            int token_id = global_index / token_offset;
            int token_bias = global_index % token_offset;
            int hi = token_bias/head_dim;
            int dim_id = token_bias%head_dim;
            if (hi < num_head + kv_num_head){
                float vec[kElemt];
                float cos_vec[kHalfElemt];
                float sin_vec[kHalfElemt];
                //1.我要找我需要取的 qkv的head_dim 
                
                float4 tmp = *reinterpret_cast<float4*>(&qkv[global_index]);
                vec[0] = tmp.x;
                vec[1] = tmp.y;
                vec[2] = tmp.z;
                vec[3] = tmp.w;
                //2.cos和sin的 head_dim/2
                int kv_len = positions[token_id];
                int emb_id = kv_len * head_dim/2 + threadIdx.x * kHalfElemt;
                float2 cos_tmp = *reinterpret_cast<const float2*>(&cos_emb[emb_id]);
                float2 sin_tmp = *reinterpret_cast<const float2*>(&sin_emb[emb_id]);
                cos_vec[0]=cos_tmp.x;
                cos_vec[1]=cos_tmp.y;
                sin_vec[0]=sin_tmp.x;
                sin_vec[1]=sin_tmp.y;
#pragma unroll
                for(int i=0;i<kHalfElemt;i++){
                    const float input_left = vec[2*i];
                    const float input_right = vec[2*i+1];
                    const float cos_now = cos_vec[i];
                    const float sin_now = sin_vec[i];
                    vec[2*i] = input_left*cos_now - input_right*sin_now;
                    vec[2*i+1] = input_left*sin_now + input_right*cos_now;
                }
                float4 res;
                res.x = vec[0];
                res.y = vec[1];
                res.z = vec[2];
                res.w = vec[3];
                *reinterpret_cast<float4*>(&qkv[global_index]) = res;
           }            
        }
    }
torch::Tensor decode_apply_rope(torch::Tensor qkv, torch::Tensor cos_emb, torch::Tensor sin_emb,torch::Tensor positions,int num_head,int kv_num_head,int head_dim) {
    TORCH_CHECK(qkv.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(qkv.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(qkv.dim() == 3, "qkv");
    TORCH_CHECK(head_dim == 128, "仅支持 head_dim=128的情况");

    int token_len = qkv.size(0);
    int all_head = num_head + 2*kv_num_head;
    dim3 block;
    block.x = kWarpSize;
    block.y = kThreads / kWarpSize;
    dim3 grid;
    int block_nums = ((token_len * all_head) + (kThreads / kWarpSize) -1 )/ (kThreads / kWarpSize);
    grid.x = min(65536,block_nums);
    //((token_len * all_head) + (kThreads / kWarpSize) -1 )/ (kThreads / kWarpSize)
    //我设置每个block处理 4个warp 每个warp 处理一个完整的 head_dim，为了防止超 我设置了个最大值
    int64_t elem_nums = token_len * (num_head + 2 * kv_num_head) * head_dim;
    // torch::Tensor output = torch::zeros_like(input);
    
    decode_apply_rope_kernel<<<grid, block>>>(
        qkv.data_ptr<float>(),
        cos_emb.data_ptr<float>(),
        sin_emb.data_ptr<float>(),
        positions.data_ptr<int>(),
        num_head,
        kv_num_head,
        head_dim,
        elem_nums);

    return qkv;
}
