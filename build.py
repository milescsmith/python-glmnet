from setuptools import Extension, setup

ext_modules = [
    Extension(
        name="_gmlnet", 
        sources=["src/glmnet/_glmnet/glmnet5.f"],
        language="fortran"
        )
    ]

def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs.update(ext_modules=ext_modules)

def pdm_build_initialize(context):
    metadata = context.config.metadata
    metadata["dependencies"].append("numpy")
    
setup(
    ext_package='_glmnet',
    ext_modules=ext_modules
)