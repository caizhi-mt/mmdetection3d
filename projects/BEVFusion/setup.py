import os
from setuptools import setup

import torch
import torch_musa
if os.getenv('FORCE_MUSA', '0') == '1':
    from torch_musa.utils.musa_extension import BuildExtension
else:
    from torch.utils.cpp_extension import BuildExtension
    from torch.utils.cpp_extension import CppExtension, CUDAExtension


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['mcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
        sources += sources_cuda
    elif os.getenv('FORCE_MUSA', '0') == '1':
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        from torch_musa.utils.musa_extension import MUSAExtension
        sources += sources_cuda
        sources = [s.replace('src', 'src_musa') for s in sources]
        sources_new = []
        for source in sources:
            if source.endswith('.cu'):
                source_new = source.split('.')[0] + '.mu'
            else:
                source_new = source
            sources_new.append(source_new)
        sources = sources_new
        extra_include_path = [s.replace('include', 'include_musa') for s in extra_include_path]

        from torch_musa.testing import get_musa_arch
        define_macros += [('MMCV_WITH_MUSA', None),
                            ('MUSA_ARCH', str(get_musa_arch()))]
        os.environ['MUSA_ARCH'] = str(get_musa_arch())
        extension = MUSAExtension
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == '__main__':
    setup(
        name='bev_pool',
        ext_modules=[
            make_cuda_ext(
                name='bev_pool_ext',
                module='projects.BEVFusion.bevfusion.ops.bev_pool',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ],
            ),
            make_cuda_ext(
                name='voxel_layer',
                module='projects.BEVFusion.bevfusion.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
