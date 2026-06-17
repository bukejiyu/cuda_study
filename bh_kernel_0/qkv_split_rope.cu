#include <torch/extension.h>
#include <vector>

namespace {
constexpr int kElemt = 128/32;
}


__global__ void qkv_split_rope_kernel(
    const float* qkv,
    const float* cos_emb,//[1,max_seq_len,1,head_dim/2]
    const float* sin_emb,
    const int* batch_per_token,//[token_len]
    const int* cu_seq_len,//[bs+1]
    int num_head,
    int kv_num_head,
    int head_dim,
    int token_len,
    int all_elemt,
    float* q,
    float* k,
    float* v){
    int global_hi = blockIdx.x * blockDim.y + threadIdx.y;
    int all_head = token_len*(num_head+2*kv_num_head);
    int stride_head = gridDim.x * blockDim.y;
    // int head_dim = 128;
    for(int h_i=global_hi;h_i<all_head;h_i+=stride_head){
        int token_id = h_i/(num_head+2*kv_num_head);
        int batch_id = batch_per_token[token_id];
        int seq_id = token_id-cu_seq_len[batch_id];
        int head_id = h_i%(num_head+2*kv_num_head);
        //qkv的 head_dim偏移位置
        int global_index = token_id * (num_head + 2*kv_num_head) * head_dim + head_id * head_dim + threadIdx.x * kElemt;
        int dim_id = global_index%head_dim;

        // embeading的偏移位置 
        // 找到embeading的偏移位置有2点 
        // 1.知道当前的token 的最大长度,比如chunked_prefill还需要 知道cachekv的长度 
        // 2.根据当前的dim_id 找到对应的 cos和sin
        float qkv_vec[kElemt];
        float cos_emb_vec[kElemt/2];
        float sin_emb_vec[kElemt/2];
        float4 tmp_qkv = *reinterpret_cast<const float4*>(&qkv[global_index]);
        qkv_vec[0]=tmp_qkv.x;
        qkv_vec[1]=tmp_qkv.y;
        qkv_vec[2]=tmp_qkv.z;
        qkv_vec[3]=tmp_qkv.w;
        int emb_id = seq_id * (head_dim)/2 + dim_id/2;
        float2 tmp_cos = *reinterpret_cast<const float2*>(&cos_emb[emb_id]);
        float2 tmp_sin = *reinterpret_cast<const float2*>(&sin_emb[emb_id]);
        cos_emb_vec[0] = tmp_cos.x;
        cos_emb_vec[1] = tmp_cos.y;
        sin_emb_vec[0] = tmp_sin.x;
        sin_emb_vec[1] = tmp_sin.y;
        if (head_id<num_head+kv_num_head){
            for(int i=0;i<kElemt/2;i++){
                float input_left = qkv_vec[2*i];
                float input_right = qkv_vec[2*i+1];
                qkv_vec[2*i] =  input_left*cos_emb_vec[i] - input_right*sin_emb_vec[i];
                qkv_vec[2*i+1] = input_left*sin_emb_vec[i] + input_right*cos_emb_vec[i];
            }
        }
        float4 res ;
        res.x = qkv_vec[0];
        res.y = qkv_vec[1];
        res.z = qkv_vec[2];
        res.w = qkv_vec[3];
        float* output;
        int output_id;
        if(head_id<num_head){
            output = q;
            output_id = token_id * num_head * head_dim + head_id * head_dim + threadIdx.x * kElemt;
        }else if(head_id<num_head+kv_num_head){
           output = k;
           output_id = token_id * kv_num_head * head_dim + (head_id-num_head) * head_dim + threadIdx.x * kElemt;
        }else{
            output = v;
            output_id = token_id * kv_num_head * head_dim + (head_id-num_head-kv_num_head) * head_dim + threadIdx.x * kElemt;
        }
        *reinterpret_cast<float4*>(output+output_id)=res;
    }

}

std::vector<torch::Tensor> qkv_split_rope(torch::Tensor qkv, torch::Tensor cos_emb, torch::Tensor sin_emb,torch::Tensor batch_per_token,torch::Tensor cu_seq_len,int num_head,int kv_num_head,int head_dim) {
    TORCH_CHECK(qkv.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(qkv.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(qkv.dim() == 3, "qkv");
    TORCH_CHECK(head_dim == 128, "仅支持 head_dim=128的情况");
    int token_len = qkv.size(0);
    int all_elemt = token_len*(num_head+2*kv_num_head)*head_dim;
    int all_heads = token_len*(num_head+2*kv_num_head);
    dim3 block(32,4);
    int block_nums = (all_heads + 4 -1)/4;
    dim3 grid(min(all_heads,65535));
    auto q = torch::zeros({token_len,num_head,head_dim}, qkv.options());
    auto k = torch::zeros({token_len,kv_num_head,head_dim}, qkv.options());
    auto v = torch::zeros({token_len,kv_num_head,head_dim}, qkv.options());

    qkv_split_rope_kernel<<<grid,block>>>(
        qkv.data_ptr<float>(),
        cos_emb.data_ptr<float>(),
        sin_emb.data_ptr<float>(),
        batch_per_token.data_ptr<int>(),
        cu_seq_len.data_ptr<int>(),
        num_head,
        kv_num_head,
        head_dim,
        token_len,
        all_elemt,
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
    );
    return {q,k,v};
}