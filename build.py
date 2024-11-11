from setuptools import Extension, setup

ext_modules = [
    Extension(
        name="fgmlnet", 
        sources=["src/glmnet/fglmnet/glmnet.f90"],
        language="fortran"
        )
    ]

def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs.update(ext_modules=ext_modules)

def pdm_build_initialize(context):
    metadata = context.config.metadata
    metadata["dependencies"].append("numpy")
    
setup(
    ext_package='fglmnet',
    ext_modules=ext_modules
)