from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 设置CUDA扩展模块的构建配置
setup(
    name='quadsim_cuda',  # 扩展模块名称
    ext_modules=[
        CUDAExtension('quadsim_cuda', [  # 定义CUDA扩展
            'quadsim.cpp',  # C++接口文件
            'quadsim_kernel.cu',  # CUDA渲染内核
            'dynamics_kernel.cu',  # CUDA动力学内核
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension  # 使用PyTorch的构建扩展命令
    })
