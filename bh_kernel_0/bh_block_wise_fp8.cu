// [fix] 添加头文件：cuda_fp8.h 提供 __nv_fp8_e4m3 类型
#include <cuda_fp8.h>
#include <torch/extension.h>

constexpr int per_thread_elemt = 4;
// [fix] 原: float epsilon =1e-10  ← 两处错误：
//   1. 缺分号
//   2. file-scope 非 constexpr float 在 device 代码中不可直接访问
//      改为 constexpr，host/device 均可见
// float epsilon =1e-10
static constexpr float epsilon = 1e-10f;

__device__ float warp_max(float x){
    for(int offset = 16; offset > 0; offset /= 2){
        float tmp = __shfl_xor_sync(0xFFFFFFFF,x,offset);
        x = max(tmp,x);
    }
    return x;
}

// [fix] 原签名中 float8_e4m3fn 未定义，改为 __nv_fp8_e4m3（来自 cuda_fp8.h）
// __global__ void per_block_fp8(const float* x,float8_e4m3fn* x_quant,float* x_scale,int block_size,int token_num,int hidden_size,int hidden_size_scale){
__global__ void per_block_fp8(const float* x, __nv_fp8_e4m3* x_quant, float* x_scale,
                               int block_size, int token_num, int hidden_size, int hidden_size_scale){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float x_vec[per_thread_elemt];
    // [fix] 原: float x_vec_fp32[per_thread_elemt]; ← 声明后从未使用，删除
    // float x_vec_fp32[per_thread_elemt];
    // [fix] 原: float8_e4m3fn res_vec[...] ← 类型未定义，改为 __nv_fp8_e4m3
    // float8_e4m3fn res_vec[per_thread_elemt];
    __nv_fp8_e4m3 res_vec[per_thread_elemt];
    // [fix] 原: int stride = blockDim.x * gridDim.x; ← 声明后从未使用，删除
    // int stride = blockDim.x * gridDim.x;
    int warp_id = tid/32;
    int num_warp = blockDim.x/32;
    int lane_id = tid%32;
    static constexpr float MAX_VALUE = 448.f;
    int num_iters = hidden_size / block_size;
    for(int token_id = blockIdx.x; token_id < token_num; token_id += gridDim.x){
        int base_id = token_id * hidden_size;
        int scale_id = token_id * hidden_size_scale;
        for(int iters = warp_id; iters < num_iters; iters += num_warp){
            int offsert_id = base_id + iters*128 + lane_id*per_thread_elemt;
            // [fix] 原: 缺分号
            // float4 input_x= *reinterpret_cast<const float4 *>(&x[offsert_id])
            float4 input_x = *reinterpret_cast<const float4*>(&x[offsert_id]);
            float max_value_thread = -5e4;
            x_vec[0] = input_x.x;
            x_vec[1] = input_x.y;
            x_vec[2] = input_x.z;
            x_vec[3] = input_x.w;
#pragma unroll
            for (int vid = 0; vid < per_thread_elemt; vid++) {
                // [fix] 原: x_vec[vid] = static_cast<float>(x_vec[vid]); ← x_vec 已是 float[]，转换无意义，删除
                // x_vec[vid] = static_cast<float>(x_vec[vid]);
                // [fix] 原: abs() ← 对 float 应用 C 整数版 abs，改为 fabsf()
                // max_value_thread = max(abs(x_vec[vid]), max_value_thread);
                max_value_thread = max(fabsf(x_vec[vid]), max_value_thread);
            }
            max_value_thread = warp_max(max_value_thread);
            max_value_thread = max(max_value_thread, epsilon);
            float scale_to_store = max_value_thread / MAX_VALUE;

#pragma unroll
            for (int vid = 0; vid < per_thread_elemt; vid++) {
                // [fix] 原: static_cast<float8_e4m3fn>(...) ← 类型未定义，改为 __nv_fp8_e4m3 构造函数
                // res_vec[vid] = static_cast<float8_e4m3fn>(x_vec[vid] * MAX_VALUE / max_value_thread);
                res_vec[vid] = __nv_fp8_e4m3(x_vec[vid] * MAX_VALUE / max_value_thread);
                x_quant[offsert_id+vid] = res_vec[vid];
            }
            if(lane_id == 0){
                int scale_now_id = scale_id + iters;
                x_scale[scale_now_id] = scale_to_store;
            }
        }
    }
}

// [fix] 原函数名 qkv_split_rope 与此 kernel 功能无关（这是 fp8 量化），重命名为 block_wise_fp8_quant
// [fix] 原 host 代码大量引用了 rope kernel 的无关变量 (qkv/cos_emb/sin_emb/q/k/v 等)，全部替换为正确参数
// std::vector<torch::Tensor> qkv_split_rope(torch::Tensor x, int block_size) {
std::vector<torch::Tensor> block_wise_fp8_quant(torch::Tensor x, int block_size) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");

    int token_num = x.size(0);
    int hidden_size = x.size(1);
    const int hidden_size_scale = hidden_size / block_size;

    // [fix] 原: GetEmptyTensor 未定义，改为 torch::empty；输出类型用 torch::kFloat8_e4m3fn
    // auto quanted_x = GetEmptyTensor({token_len, hidden_size}, x.options());
    auto x_quant = torch::empty({token_num, hidden_size},
                                x.options().dtype(torch::kFloat8_e4m3fn));
    // [fix] 原: qkv.options() 变量不存在，改为 x.options()
    // auto x_scale = torch::zeros({token_len,hidden_size_scale}, qkv.options());
    auto x_scale = torch::zeros({token_num, hidden_size_scale}, x.options());

    dim3 grid(min(132 * 8, token_num));
    dim3 block(min(1024, hidden_size / block_size * 32));

    // [fix] 原 kernel 调用参数全部错误（传的是 rope 的参数），改为正确参数
    per_block_fp8<<<grid, block>>>(
        x.data_ptr<float>(),
        reinterpret_cast<__nv_fp8_e4m3*>(x_quant.data_ptr()),
        x_scale.data_ptr<float>(),
        block_size,
        token_num,
        hidden_size,
        hidden_size_scale
    );
    return {x_quant, x_scale};
}
