# SPDX-License-Identifier: MIT
# Compiler and internal Vulkan pipeline gate; no installed ICD.
add_languages('rust', required : true)
apex_rust = import('rust')
apex_rs = static_library('apex_rs', 'lib.rs', rust_abi : 'c')
apex_nir_algebraic_c = custom_target('apex_nir_algebraic.c',
  input : 'apex_nir_algebraic.py', output : 'apex_nir_algebraic.c',
  command : [prog_python, '@INPUT@', '-p', dir_compiler_nir],
  capture : true, depend_files : nir_algebraic_depends,
)
apex_nir = static_library('apex_nir', ['from_nir.c', apex_nir_algebraic_c],
  include_directories : [inc_include, inc_src, inc_nir],
  dependencies : [idep_nir, idep_mesautil],
)
apex_spirv = static_library('apex_spirv', ['spirv.c'],
  include_directories : [inc_include, inc_src],
  dependencies : [idep_vtn, idep_nir, idep_mesautil, dependency('SPIRV-Tools')],
)
executable('apex-compile', 'main.c',
  include_directories : [inc_include, inc_src],
  dependencies : [idep_mesautil],
  link_with : [apex_spirv, apex_nir, apex_rs],
  link_language : 'cpp',
  install : false,
)
# The tests run programs on the ISA reference model, Tooling/apex_isa.
apex_isa = get_option('apex-isa')
if apex_isa == ''
  apex_isa = meson.project_source_root() / '..' / '..' / 'Tooling'
endif
apex_test_env = {'APEX_ISA' : apex_isa}
apex_rust.test('apex-native', apex_rs, suite : ['apex'], env : apex_test_env)
apex_compiler_test = executable('apex-compiler-test', 'compiler_test.c',
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nir, idep_mesautil],
  link_with : [apex_nir, apex_rs],
  link_language : 'cpp',
  install : false,
)
test('apex-compiler', apex_compiler_test, suite : ['apex'], env : apex_test_env)
# Benchmark, triangle and compositor shaders plus the captured GNOME Shell
# and glmark2 shaders run on the ISA model under the zero-spill gate.
apex_test_shaders = []
foreach s : ['vulkan/bench.comp', 'vulkan/bench.vert', 'vulkan/bench.frag',
             'vulkan/triangle.vert', 'vulkan/triangle.frag', 'tests/compositor.frag',
             'tests/features.comp', 'tests/spill.comp', 'tests/waterfall.comp', 'tests/control.comp', 'tests/texture.comp', 'tests/discard.frag', 'tests/demote.frag', 'tests/depth.frag',
             'tests/transform.vert', 'tests/mix.frag', 'tests/hoist.frag']
  apex_test_shaders += custom_target('apex-test-' + s.underscorify(),
    input : s, output : '@PLAINNAME@.spv',
    command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1',
               '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default : true,
  )
endforeach
apex_shader_test = executable('apex-shader-test', 'shader_test.c',
  include_directories : [inc_include, inc_src],
  dependencies : [idep_mesautil, dep_m],
  link_with : [apex_spirv, apex_nir, apex_rs],
  link_language : 'cpp',
  install : false,
)
test('apex-shaders', apex_shader_test,
  args : [meson.current_build_dir(), meson.current_source_dir() / 'tests/gnome',
          meson.current_source_dir() / 'tests/glmark2'],
  depends : apex_test_shaders, suite : ['apex'], env : apex_test_env, timeout : 600)
subdir('vulkan')
