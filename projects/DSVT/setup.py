import os
from setuptools import setup

import torch
from torch_musa.utils.musa_extension import (BuildExtension, CppExtension,
                                       MUSAExtension)


def make_musa_ext(name,
                  module,
                  sources,
                  sources_musa=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.musa.is_available() or os.getenv('FORCE_MUSA', '0') == '1':
        define_macros += [('WITH_MUSA', None)]
        extension = MUSAExtension
        extra_compile_args['mcc'] = extra_args + [
            '-D__MUSA_NO_HALF_OPERATORS__',
            '-D__MUSA_NO_HALF_CONVERSIONS__',
            '-D__MUSA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
        sources += sources_musa
    else:
        print('Compiling {} without MUSA'.format(name))
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
        name='dsvt',
        ext_modules=[
            make_musa_ext(
                name='ingroup_inds_musa',
                module='projects.DSVT.dsvt.ops.ingroup_inds',
                sources=[
                    'src/ingroup_inds.cpp',
                    'src/ingroup_inds_kernel.cu',
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
