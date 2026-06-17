from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bh_ops",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "bh_ops._C",
            ["bh_kernel_0/bindings.cpp", "bh_kernel_0/bh_prefixsum_1.cu", "bh_kernel_0/bh_softmax.cu", "bh_kernel_0/bh_reduce_sum.cu", "bh_kernel_0/bh_transpose.cu", "bh_kernel_0/bh_group_norm.cu", "bh_kernel_0/bh_group_norm_0.cu", "bh_kernel_0/bh_rope.cu", "bh_kernel_0/bh_rope0.cu", "bh_kernel_0/bh_rope_1.cu", "bh_kernel_0/bh_gemm.cu", "bh_kernel_0/bh_block_wise_fp8.cu", "bh_kernel_0/qkv_split_rope.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["bh_ops"],
    package_dir={"bh_ops": "bh_ops_py"},
)
