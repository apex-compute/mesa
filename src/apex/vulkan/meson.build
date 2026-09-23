# SPDX-License-Identifier: MIT
apex_pipeline = static_library('apex_pipeline', ['apex_pipeline.c', 'apex_device.c'],
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime_headers, idep_nir],
)
apex_pipeline_spv = custom_target('apex-pipeline-spv',
  input : 'pipeline.spvasm', output : 'pipeline.spv',
  command : [find_program('spirv-as'), '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_descriptors_spv = custom_target('apex-descriptors-spv',
  input : 'descriptors.comp', output : 'descriptors.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_atomics_spv = custom_target('apex-atomics-spv',
  input : 'descriptor_atomics.comp', output : 'descriptor_atomics.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_reindex_spv = custom_target('apex-reindex-spv',
  input : 'descriptor_reindex.spvasm', output : 'descriptor_reindex.spv',
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
test('apex-descriptors-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_descriptors_spv], suite : ['apex'])
test('apex-atomics-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_atomics_spv], suite : ['apex'])
test('apex-reindex-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_reindex_spv], suite : ['apex'])
test('apex-pipeline', apex_pipeline_test,
  args : [apex_pipeline_spv, apex_descriptors_spv, apex_atomics_spv, apex_reindex_spv], suite : ['apex'])
apex_command_test = executable('apex-command-test', 'command_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_nir, idep_mesautil],
  link_with : [apex_pipeline, apex_nir, apex_rs],
  link_args : ['-Wl,--wrap=ioctl'],
  link_language : 'cpp', install : false,
)
test('apex-command', apex_command_test, args : [apex_pipeline_spv], suite : ['apex'])
apex_sync_test = executable('apex-sync-test', 'sync_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_nir, idep_mesautil],
  link_args : ['-Wl,--wrap=cnd_wait'],
  link_language : 'cpp', install : false,
)
test('apex-sync', apex_sync_test, suite : ['apex'])
