project('test_builds', 'c',
  version : '0.1',
  default_options : ['warning_level=3'])

add_languages('fortran')

# https://mesonbuild.com/Python-module.html
# requires atleast 0.46.0
py_mod = import('python')
py3 = py_mod.find_installation()
py3_dep = py3.dependency()

incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

py3.extension_module('_glmnet',
           'glmnet5.f',
           '_glmnet5module.c',
           'fortranobject.c',
           include_directories: inc_np,
           dependencies : py3_dep,
           install : true)
