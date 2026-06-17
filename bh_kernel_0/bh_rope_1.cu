#include <torch/extension.h>
//qkv [bs,xx,head+2kv_head,dim] -> [token_len,head+2kv_head,dim]
//cos_emb [max_seq_len,head_dim/2]
//sin_emb [max_seq_len,head_dim/2]
// positions [bs]
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kElemt = 4;
constexpr int kHalfElemt =2;

//batch_per_token[token_len]
//cu_seq_len[bs]
__global__ void prefill_apply_rope(float* qkv,const float* cos_emb,const float* sin_emb,int* batch_per_token,int* cu_seq_len,int num_head,int kv_num_head,int head_dim,int token_len){
    int all_warp = gridDim.x*blockDim.y;//可以处理的总head数
    int global_warp_id = blockIdx.x*blockDim.y+threadIdx.y;//每个head的id
    int all_head = token_len*(num_head+kv_num_head);
    int offset = (num_head+2*kv_num_head)*head_dim;
    int qk_head = num_head+kv_num_head;
    for(int global_hi = global_warp_id;global_hi < all_head; global_hi += all_warp){
        int token_id = global_hi / qk_head;
        int qk_head_id = global_hi % qk_head;
        int global_index = token_id * offset + qk_head_id * head_dim + threadIdx.x * kElemt;
        float qkv_vec[kElemt];
        float4 qkv_float4 = *reinterpret_cast<float4*>(&qkv[global_index]);
        qkv_vec[0]=qkv_float4.x;
        qkv_vec[1]=qkv_float4.y;
        qkv_vec[2]=qkv_float4.z;
        qkv_vec[3]=qkv_float4.w;
        int bs = batch_per_token[token_id];
        int seq_start = cu_seq_len[bs];
        int seq_now = token_id - seq_start;

        //获取 emb_id
        int emb_id = seq_now*(head_dim/2)+threadIdx.x*kHalfElemt;
        float cos_vec[kHalfElemt];
        float sin_vec[kHalfElemt];
        float2 cos_float2 = *reinterpret_cast<const float2*>(&cos_emb[emb_id]);
        float2 sin_float2 = *reinterpret_cast<const float2*>(&sin_emb[emb_id]);
        cos_vec[0] = cos_float2.x;
        cos_vec[1] = cos_float2.y;
        sin_vec[0] = sin_float2.x;
        sin_vec[1] = sin_float2.y;
        for(int i = 0;i < kHalfElemt;i++){
            float input_left = qkv_vec[2*i];
            float input_right = qkv_vec[2*i+1];
            qkv_vec[2*i]=input_left*cos_vec[i]-input_right*sin_vec[i];
            qkv_vec[2*i+1] = input_left*sin_vec[i]+input_right*cos_vec[i];
        }
        float4 res;
        res.x=qkv_vec[0];
        res.y=qkv_vec[1];
        res.z=qkv_vec[2];
        res.w=qkv_vec[3];
        *reinterpret_cast<float4*>(&qkv[global_index]) = res;
    }
}
torch::Tensor prefill_apply_rope(torch::Tensor qkv, torch::Tensor cos_emb, torch::Tensor sin_emb,torch::Tensor batch_per_token,torch::Tensor cu_seq_len,int num_head,int kv_num_head,int head_dim) {
    TORCH_CHECK(qkv.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(qkv.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(qkv.dim() == 3, "qkv");
    TORCH_CHECK(head_dim == 128, "仅支持 head_dim=128的情况");
    int token_len = qkv.size(0);
    int all_elemt = token_len*(num_head+2*kv_num_head)*head_dim;
    int all_heads = token_len*(num_head+kv_num_head);
    dim3 block(32,4);
    int block_nums = (all_heads + 4 -1)/4;
    dim3 grid(min(all_heads,65535));
    prefill_apply_rope<<<grid,block>>>(
        qkv.data_ptr<float>(),
        cos_emb.data_ptr<float>(),
        sin_emb.data_ptr<float>(),
        batch_per_token.data_ptr<int>(),
        cu_seq_len.data_ptr<int>(),
        num_head,
        kv_num_head,
        head_dim,
        token_len
    );
    return qkv;
}