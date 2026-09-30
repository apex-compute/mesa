# SPDX-License-Identifier: MIT
# Internal kernels are SPIR-V embedded as C arrays.
apex_internal_spv = []
foreach kernel : ['setup', 'bin', 'copy', 'clear', 'timestamp', 'querycopy', 'resolve', 'etc2']
  apex_internal_spv += custom_target('apex_' + kernel + '_spv',
    input : kernel + '.comp', output : 'apex_' + kernel + '_spv.h',
    command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1',
               '-I' + meson.current_source_dir(), '--vn', 'apex_' + kernel + '_spv',
               '@INPUT@', '-o', '@OUTPUT@'],
    depend_files : files('apex_draw.h'),
  )
endforeach
apex_pipeline = static_library('apex_pipeline',
  ['apex_pipeline.c', 'apex_device.c', 'apex_graphics.c', 'apex_format.c', 'apex_texture.c', apex_internal_spv],
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime_headers, idep_vulkan_wsi_headers, idep_nir],
)
apex_entrypoints = custom_target('apex_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['apex_entrypoints.h', 'apex_entrypoints.c'],
  command : [prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
             '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'apex',
             '--beta', with_vulkan_beta.to_string()],
  depend_files : vk_entrypoints_gen_depend_files,
)
# Uninstalled development driver: instance creation requires APEX_DEVELOPMENT=1.
apex_instance = static_library('apex_instance', ['apex_instance.c', apex_entrypoints],
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime_headers, idep_vulkan_wsi_headers, idep_mesautil, dep_libdrm],
)
apex_icd = shared_library('vulkan_apex',
  link_whole : [apex_instance],
  link_with : [apex_pipeline, apex_nir, apex_rs],
  dependencies : [idep_vulkan_runtime, idep_vulkan_wsi, idep_nir, idep_mesautil, dep_libdrm],
  link_args : [vulkan_icd_link_args, ld_args_bsymbolic, ld_args_gc_sections],
  link_depends : vulkan_icd_link_depends,
  link_language : 'cpp', gnu_symbol_visibility : 'hidden', install : false,
)
apex_development_icd = custom_target('apex_development_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'apex_development_icd.@0@.json'.format(host_machine.cpu()),
  command : [prog_python, '@INPUT0@', '--api-version', '1.4', '--xml', '@INPUT1@',
             '--sizeof-pointer', sizeof_pointer, '--icd-lib-path', meson.current_build_dir(),
             '--icd-filename', libname_prefix + 'vulkan_apex.' + libname_suffix,
             '--out', '@OUTPUT@'],
  build_by_default : true,
)
apex_instance_test = executable('apex-instance-test', ['instance_test.c', apex_entrypoints[0]],
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_vulkan_wsi, idep_nir, idep_mesautil, dep_libdrm, dep_dl],
  link_with : [apex_instance, apex_pipeline, apex_nir, apex_rs],
  link_args : ['-Wl,--wrap=drmGetDevices2', '-Wl,--wrap=drmFreeDevices',
               '-Wl,--wrap=drmGetVersion', '-Wl,--wrap=drmFreeVersion',
               '-Wl,--wrap=drmGetCap', '-Wl,--wrap=drmSyncobjCreate',
               '-Wl,--wrap=drmSyncobjDestroy', '-Wl,--wrap=drmSyncobjWait',
               '-Wl,--wrap=open64', '-Wl,--wrap=ioctl'],
  link_language : 'cpp', install : false,
)
test('apex-instance', apex_instance_test, args : ['--mock'], suite : ['apex'])
test('apex-loader', apex_instance_test, args : [apex_development_icd],
  depends : [apex_icd], suite : ['apex'])
# Hardware execution is explicit, never part of the ordinary host suite.
apex_application_test = executable('apex-application-test', 'application_test.c',
  include_directories : inc_include, dependencies : [dep_dl, dep_libdrm], install : false,
)
apex_external_sync_application = executable('apex-external-sync-application', 'external_sync_application.c',
  include_directories : inc_include, dependencies : dep_dl, install : false,
)
apex_application_spv = custom_target('apex-application-spv',
  input : 'pipeline.spvasm', output : 'application.spv',
  command : [find_program('spirv-as'), '--target-env', 'vulkan1.0', '@INPUT@', '-o', '@OUTPUT@'],
  build_by_default : true,
)
test('apex-application-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.0', apex_application_spv], suite : ['apex'])
apex_pipeline_spv = custom_target('apex-pipeline-spv',
  input : 'pipeline.spvasm', output : 'pipeline.spv',
  command : [find_program('spirv-as'), '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_descriptors_spv = custom_target('apex-descriptors-spv',
  input : 'descriptors.comp', output : 'descriptors.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_scalar_spv = custom_target('apex-scalar-spv',
  input : 'scalar_descriptors.comp', output : 'scalar_descriptors.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_images_spv = custom_target('apex-images-spv',
  input : 'images.comp', output : 'images.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_rgba_spv = custom_target('apex-rgba-spv',
  input : 'rgba.comp', output : 'rgba.spv',
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
apex_dispatch_spv = custom_target('apex-dispatch-spv',
  input : 'dispatch.comp', output : 'dispatch.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_multiple_spv = custom_target('apex-multiple-spv',
  input : 'multiple.comp', output : 'multiple.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
)
apex_texture_spv = custom_target('apex-texture-spv',
  input : 'texture.comp', output : 'texture.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
  build_by_default : true,
)
apex_pipeline_test = executable('apex-pipeline-test', 'pipeline_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_vulkan_wsi, idep_nir, idep_mesautil],
  link_with : [apex_pipeline, apex_nir, apex_rs],
  link_language : 'cpp', install : false,
)
test('apex-pipeline-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_pipeline_spv], suite : ['apex'])
test('apex-descriptors-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_descriptors_spv], suite : ['apex'])
test('apex-scalar-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', '--scalar-block-layout', apex_scalar_spv], suite : ['apex'])
test('apex-images-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_images_spv], suite : ['apex'])
test('apex-rgba-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_rgba_spv], suite : ['apex'])
test('apex-atomics-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_atomics_spv], suite : ['apex'])
test('apex-reindex-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_reindex_spv], suite : ['apex'])
test('apex-dispatch-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_dispatch_spv], suite : ['apex'])
test('apex-multiple-spirv', find_program('spirv-val'),
  args : ['--target-env', 'vulkan1.1', apex_multiple_spv], suite : ['apex'])
test('apex-pipeline', apex_pipeline_test,
  args : [apex_pipeline_spv, apex_descriptors_spv, apex_atomics_spv, apex_reindex_spv, apex_dispatch_spv, apex_multiple_spv, apex_scalar_spv, apex_images_spv, apex_rgba_spv], suite : ['apex'])
apex_command_test = executable('apex-command-test', 'command_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_vulkan_wsi, idep_nir, idep_mesautil],
  link_with : [apex_pipeline, apex_nir, apex_rs],
  link_args : ['-Wl,--wrap=ioctl'],
  link_language : 'cpp', install : false,
)
test('apex-command', apex_command_test, args : [apex_pipeline_spv], suite : ['apex'])
test('apex-image-command', apex_command_test, args : [apex_images_spv, '--images'], suite : ['apex'])
apex_external_memory_test = executable('apex-external-memory-test', 'external_memory_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_vulkan_wsi, idep_nir, idep_mesautil],
  link_with : [apex_pipeline, apex_nir, apex_rs],
  link_args : ['-Wl,--wrap=ioctl'],
  link_language : 'cpp', install : false,
)
test('apex-external-memory', apex_external_memory_test, suite : ['apex'])
apex_async_test = executable('apex-async-test', 'async_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_vulkan_wsi, idep_nir, idep_mesautil],
  link_with : [apex_pipeline, apex_nir, apex_rs],
  link_args : ['-Wl,--wrap=ioctl'],
  link_language : 'cpp', install : false,
)
test('apex-async', apex_async_test, args : [apex_pipeline_spv], suite : ['apex'])
apex_sync_test = executable('apex-sync-test', 'sync_test.c',
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [idep_vulkan_runtime, idep_vulkan_wsi, idep_nir, idep_mesautil],
  link_args : ['-Wl,--wrap=cnd_wait'],
  link_language : 'cpp', install : false,
)
test('apex-sync', apex_sync_test, suite : ['apex'])
# Explicit hardware numeric gates drive this runner with host references.
apex_run = executable('apex-run', 'run.c',
  include_directories : inc_include, dependencies : dep_dl, install : false,
)
# Explicit hardware graphics gate through the public loader.
apex_graphics_test = executable('apex-graphics-test', 'graphics_test.c',
  include_directories : inc_include, dependencies : [dep_dl, dep_m], install : false,
)
apex_triangle_spv = []
foreach stage : ['vert', 'frag']
  apex_triangle_spv += custom_target('apex-triangle-' + stage,
    input : 'triangle.' + stage, output : 'triangle.' + stage + '.spv',
    command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.0', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default : true,
  )
endforeach
apex_clip_spv = custom_target('apex-clip-spv',
  input : 'clip.vert', output : 'clip.vert.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.0', '@INPUT@', '-o', '@OUTPUT@'],
)
test('apex-graphics-programs', apex_pipeline_test,
  args : ['--graphics', apex_texture_spv, apex_triangle_spv, apex_clip_spv],
  env : ['NIR_DEBUG=validate'], suite : ['apex'])
# Frame-time benchmark for the performance plan's desktop and game scenes.
apex_bench = executable('apex-bench', 'bench.c',
  include_directories : inc_include, dependencies : [dep_dl, dep_m], install : false,
)
foreach stage : ['vert', 'frag', 'comp']
  custom_target('apex-bench-' + stage,
    input : 'bench.' + stage, output : 'bench.' + stage + '.spv',
    command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.0', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default : true,
  )
endforeach
apex_present_test = executable('apex-present-test', 'present_test.c',
  include_directories : inc_include, dependencies : [dep_dl, dep_m], install : false,
)
apex_texture_test = executable('apex-texture-test', 'texture_test.c',
  include_directories : inc_include, dependencies : [dep_dl, dep_m], install : false,
)
custom_target('apex-ms-texture-spv',
  input : 'ms_texture.comp', output : 'ms_texture.spv',
  command : [find_program('glslangValidator'), '-V', '--target-env', 'vulkan1.1', '@INPUT@', '-o', '@OUTPUT@'],
  build_by_default : true,
)
