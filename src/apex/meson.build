# SPDX-License-Identifier: MIT
# Compiler and internal Vulkan pipeline gate; no installed ICD.
add_languages('rust', required : true)
apex_rust = import('rust')
apex_rs = static_library('apex_rs', 'lib.rs', rust_abi : 'c')
apex_nir = static_library('apex_nir', ['from_nir.c', 'float.c'],
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nir, idep_mesautil],
)
executable('apex-compile', 'main.c',
  include_directories : [inc_include, inc_src],
  dependencies : [idep_vtn, idep_nir, idep_mesautil, dependency('SPIRV-Tools')],
  link_with : [apex_nir, apex_rs],
  link_language : 'cpp',
  install : false,
)
apex_rust.test('apex-native', apex_rs, suite : ['apex'])
apex_compiler_test = executable('apex-compiler-test', 'compiler_test.c',
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nir, idep_mesautil],
  link_with : [apex_nir, apex_rs],
  link_language : 'cpp',
  install : false,
)
test('apex-compiler', apex_compiler_test, suite : ['apex'])
subdir('vulkan')
