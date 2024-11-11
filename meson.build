# This is going to be a Frankenstein's monster of
# a build file based on that from https://github.com/amolenaar/meson-python-pdm-example
# and Scipy
project('glmnet', 'c',
  version : '0.1',
  default_options : [
    'warning_level=1',
    'fortran_std=legacy',
    'buildtype=debugoptimized',
    ])

# https://mesonbuild.com/Python-module.html
# requires atleast 0.46.0
py_mod = import('python')
py3 = py_mod.find_installation()
py3_dep = py3.dependency()

add_languages('fortran', native: false)
ff = meson.get_compiler('fortran')
if ff.has_argument('-Wno-conversion')
  add_project_arguments('-Wno-conversion', language: 'fortran')
endif

_global_ff_args = ff.get_supported_arguments(
  '-fallow-argument-mismatch',
  '-fdefault-real-8',
  '-fdefault-double-8',
  '-fdefault-integer-8',
  '-ffree-line-length-none',
  '-fbacktrace',
  '-w',
)
add_project_arguments(_global_ff_args, language: 'fortran')

incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _incdir_numpy_abs = run_command(py3,
    [
      '-c',
      '''import os
os.chdir("..")
import numpy
print(numpy.get_include())
      '''
    ],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif

inc_np = include_directories(incdir_numpy)
incdir_f2py = run_command(py3,
  [
  '-c',
  'import os; import numpy as np; incdir = os.path.relpath(np.f2py.get_include()); print(incdir)',
  ],
  check: true
).stdout().strip()

inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  # c_args: numpy_nodepr_api,
  dependencies: py3_dep,
  include_directories: [inc_np, inc_f2py],
  gnu_symbol_visibility: 'hidden',
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [inc_np, inc_f2py],
)

subdir('src/glmnet')