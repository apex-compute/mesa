# SPDX-License-Identifier: MIT
apex_pipeline = static_library('apex_pipeline', 'apex_pipeline.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime_headers, idep_nir],
)
apex_pipeline_spv = custom_target('apex-pipeline-spv',
  input : 'pipeline.spvasm', output : 'pipeline.spv',
  command : [find_program('spirv-as'), '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_pipeline_test = executable('apex-pipeline-test', 'pipeline_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_nir, idep_mesautil],
  link_with : [apex_pipeline, apex_nir, apex_rs],
  link_language : 'cpp', install : false,
)
test('apex-pipeline-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_pipeline_spv], suite : ['apex'])
test('apex-pipeline', apex_pipeline_test, args : [apex_pipeline_spv], suite : ['apex'])
