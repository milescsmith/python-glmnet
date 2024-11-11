py3.install_sources(
  [
    '__init__.py'
  ],
  pure : false,
  subdir : 'src/glmnet/fglmnet'
)

py3.extension_module(
  'fglmnet',
  [
    'glmnet5.f',
    'fglmnetmodule.c',
    'fglmnet-f2pywrappers.f',
  ],
  include_directories: inc_np,
  dependencies: [py3_dep, fortranobject_dep],
  link_with: fortranobject_lib,
  install : true,
  # subdir: 'src/glmnet/fglmnet'
)