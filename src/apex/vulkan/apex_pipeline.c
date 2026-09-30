/* SPDX-License-Identifier: MIT */
#include "apex_pipeline.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_device.h"
#include "vk_log.h"
#include "util/log.h"
#include "util/os_misc.h"
#include "util/u_atomic.h"
#include <stdio.h>

struct descriptor_lowering {
   struct apex_program *program;
   bool invalid;
};

static bool
lower_resource(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   nir_def *replacement;
   b->cursor = nir_before_instr(&i->instr);
   switch (i->intrinsic) {
   case nir_intrinsic_vulkan_resource_index: {
      unsigned set = nir_intrinsic_desc_set(i), binding = nir_intrinsic_binding(i);
      if (set >= ctx->program->set_count ||
          (nir_intrinsic_desc_type(i) != nir_descriptor_type_storage_buffer &&
           nir_intrinsic_desc_type(i) != nir_descriptor_type_uniform_buffer)) {
         ctx->invalid = true;
         return false;
      }
      const struct apex_set_layout *layout =
         ctx->program->set_layouts[set];
      if (!layout || binding >= layout->binding_count || !layout->bindings[binding].count) {
         ctx->invalid = true;
         return false;
      }
      VkDescriptorType type = nir_intrinsic_desc_type(i) == nir_descriptor_type_uniform_buffer ?
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      VkDescriptorType dynamic_type = nir_intrinsic_desc_type(i) == nir_descriptor_type_uniform_buffer ?
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      if (layout->bindings[binding].type != type && layout->bindings[binding].type != dynamic_type &&
          (type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
           layout->bindings[binding].type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)) {
         ctx->invalid = true;
         return false;
      }
      nir_def *index = i->src[0].ssa;
      BITSET_SET_COUNT(ctx->program->used_descriptors,
         ctx->program->set_offsets[set] + layout->bindings[binding].slot,
         layout->bindings[binding].count);
      /* Opaque resource references retain the binding bounds through reindex.
       * The table is limited to 4096 entries, so base/count fit in 16 bits. */
      unsigned base = ctx->program->set_offsets[set] + layout->bindings[binding].slot;
      unsigned packed = base | (layout->bindings[binding].count << 16);
      replacement = nir_vec2(b, nir_imm_int(b, packed), index);
      break;
   }
   case nir_intrinsic_vulkan_resource_reindex:
      replacement = nir_vec2(b, nir_channel(b, i->src[0].ssa, 0),
         nir_iadd(b, nir_channel(b, i->src[0].ssa, 1), i->src[1].ssa));
      break;
   case nir_intrinsic_load_vulkan_descriptor: {
      nir_def *packed = nir_channel(b, i->src[0].ssa, 0);
      nir_def *index = nir_channel(b, i->src[0].ssa, 1);
      nir_def *slot = nir_bcsel(b, nir_ult(b, index, nir_ushr_imm(b, packed, 16)),
         nir_iadd(b, nir_iand_imm(b, packed, 0xffff), index),
         nir_imm_int(b, ctx->program->descriptor_count));
      /* The descriptor load changes the opaque reference into index/offset. */
      replacement = nir_vec2(b, slot, nir_imm_int(b, 0));
      break;
   }
   default:
      return false;
   }
   nir_def_rewrite_uses(&i->def, replacement);
   nir_instr_remove(&i->instr);
   return true;
}

/* Accesses component c of a lowered buffer operation at start. */
static nir_def *
buffer_component(nir_builder *b, nir_intrinsic_instr *i, nir_def *start, unsigned c)
{
   nir_def *address = nir_build_addr_iadd_imm(b, start, nir_address_format_2x32bit_global,
                                              nir_var_mem_global, c * 4);
   switch (i->intrinsic) {
   case nir_intrinsic_store_ssbo:
      nir_store_global_2x32(b, nir_channel(b, i->src[0].ssa, c), address,
                            .align_mul = 4, .access = nir_intrinsic_access(i));
      return NULL;
   case nir_intrinsic_ssbo_atomic_swap:
      return nir_global_atomic_swap_2x32(b, 32, address, i->src[2].ssa, i->src[3].ssa,
         .atomic_op = nir_intrinsic_atomic_op(i), .access = nir_intrinsic_access(i));
   case nir_intrinsic_ssbo_atomic:
      return nir_global_atomic_2x32(b, 32, address, i->src[2].ssa,
         .atomic_op = nir_intrinsic_atomic_op(i), .access = nir_intrinsic_access(i));
   default:
      return nir_load_global_2x32(b, 1, 32, address, .align_mul = 4, .access = nir_intrinsic_access(i));
   }
}

/* Whether bytes offset..offset + width fit in range. Subtract from the range
 * before comparing: offset + width may wrap. */
static nir_def *
buffer_inside(nir_builder *b, nir_def *range, nir_def *offset, unsigned width)
{
   return nir_iand(b, nir_uge_imm(b, range, width),
                   nir_uge(b, nir_iadd_imm(b, range, -(int)width), offset));
}

static bool
lower_buffer(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   bool store = i->intrinsic == nir_intrinsic_store_ssbo;
   bool size = i->intrinsic == nir_intrinsic_get_ssbo_size;
   bool swap = i->intrinsic == nir_intrinsic_ssbo_atomic_swap;
   bool atomic = swap || i->intrinsic == nir_intrinsic_ssbo_atomic;
   if (atomic && nir_intrinsic_offset_shift(i)) {
      ctx->invalid = true;
      return false;
   }
   if (!store && !size && !atomic && i->intrinsic != nir_intrinsic_load_ssbo &&
       i->intrinsic != nir_intrinsic_load_ubo)
      return false;
   if (!size && ((store ? i->src[0].ssa->bit_size : i->def.bit_size) != 32 ||
                 i->num_components > 4 ||
                 (atomic ? i->num_components != 1 :
                  nir_intrinsic_align_mul(i) < 4 || nir_intrinsic_align_offset(i) % 4))) {
      ctx->invalid = true;
      return false;
   }
   b->cursor = nir_before_instr(&i->instr);
   nir_def *row = nir_imul_imm(b, i->src[store ? 1 : 0].ssa, sizeof(union apex_descriptor));
   nir_def *words[3];
   /* Submission metadata is immutable; every slot, including the null
    * sentinel, has backing independent of the application's resource range. */
   for (unsigned c = 0; c < 3; c++)
      words[c] = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), nir_iadd_imm(b, row, c * 4),
         .align_mul = 4, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   if (size) {
      nir_def_rewrite_uses(&i->def, words[2]);
      nir_instr_remove(&i->instr);
      return true;
   }
   nir_def *offset = i->src[store ? 2 : 1].ssa;
   nir_def *start = nir_build_addr_iadd(b, nir_vec2(b, words[0], words[1]),
      nir_address_format_2x32bit_global, nir_var_mem_global, offset);
   unsigned mask = store ? nir_intrinsic_write_mask(i) : BITFIELD_MASK(i->num_components);
   unsigned count = util_last_bit(mask);
   nir_def *whole[4], *values[4];
   /* A vector wholly inside the range takes one branch-free path. Others,
    * including the zero-range null sentinel, bound each component. */
   if (count > 1) {
      nir_push_if(b, buffer_inside(b, words[2], offset, count * 4));
      u_foreach_bit(c, mask)
         whole[c] = buffer_component(b, i, start, c);
      nir_push_else(b, NULL);
   }
   u_foreach_bit(c, mask) {
      nir_push_if(b, buffer_inside(b, words[2], offset, c * 4 + 4));
      nir_def *loaded = buffer_component(b, i, start, c);
      nir_push_else(b, NULL);
      nir_def *zero = nir_imm_int(b, 0);
      nir_pop_if(b, NULL);
      if (!store)
         values[c] = nir_if_phi(b, loaded, zero);
   }
   if (count > 1) {
      nir_pop_if(b, NULL);
      if (!store)
         for (unsigned c = 0; c < count; c++)
            values[c] = nir_if_phi(b, whole[c], values[c]);
   }
   if (!store)
      nir_def_rewrite_uses(&i->def, nir_vec(b, values, i->num_components));
   nir_instr_remove(&i->instr);
   return true;
}

/* Image memory is global memory: storage images lower to global access in
 * apex_lower_textures. */
static bool
lower_image(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   if (i->intrinsic != nir_intrinsic_barrier)
      return false;
   nir_variable_mode modes = nir_intrinsic_memory_modes(i);
   if (!(modes & nir_var_image))
      return false;
   nir_intrinsic_set_memory_modes(i, (modes & ~nir_var_image) | nir_var_mem_global);
   return true;
}

void
apex_program_finish(struct apex_device *device, struct apex_program *program)
{
   if (program->bo.handle)
      apex_bo_finish(device, &program->bo);
   apex_compile_result_finish(&program->code);
}

static void
apex_pipeline_destroy(struct vk_device *device, struct vk_pipeline *vk,
                      const VkAllocationCallbacks *alloc)
{
   struct apex_pipeline *pipeline = (struct apex_pipeline *)vk;
   apex_program_finish((struct apex_device *)device, &pipeline->program);
   if (pipeline->layout)
      vk_pipeline_layout_unref(device, pipeline->layout);
   vk_pipeline_free(device, alloc, vk);
}

struct workgroup_lowering {
   nir_variable *id, *grid;
   nir_intrinsic_instr *native;
};

static bool
lower_workgroup(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct workgroup_lowering *ctx = data;
   nir_def *replacement;
   b->cursor = nir_before_instr(&i->instr);
   switch (i->intrinsic) {
   case nir_intrinsic_load_num_workgroups:
      replacement = nir_load_var(b, ctx->grid);
      break;
   case nir_intrinsic_load_workgroup_id:
      replacement = nir_load_var(b, ctx->id);
      break;
   case nir_intrinsic_load_base_workgroup_id:
      /* SPIR-V lowering added the native base to WorkgroupID; the linear
       * workgroup already includes it. */
      if (i == ctx->native)
         return false;
      replacement = nir_imm_zero(b, 3, 32);
      break;
   case nir_intrinsic_load_global_invocation_id: {
      nir_def *size = nir_imm_ivec3(b, b->shader->info.workgroup_size[0],
         b->shader->info.workgroup_size[1], b->shader->info.workgroup_size[2]);
      replacement = nir_iadd(b, nir_imul(b, nir_load_var(b, ctx->id), size),
                             nir_load_local_invocation_id(b));
      break;
   }
   default:
      return false;
   }
   nir_def_rewrite_uses(&i->def, replacement);
   nir_instr_remove(&i->instr);
   return true;
}

/* Each native workgroup runs linear workgroups first + native, stepping by
 * the stride below end. Indirect jobs read the grid from the
 * VkDispatchIndirectCommand and cover all of it. */
static void
wrap_workgroups(const struct apex_program *program, nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct workgroup_lowering ctx = {
      .id = nir_local_variable_create(impl, glsl_uvec_type(3), "workgroup"),
      .grid = nir_local_variable_create(impl, glsl_uvec_type(3), "grid"),
   };
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder;
   unsigned trailer = apex_program_trailer(program);
   nir_def *words[11];
   for (unsigned w = 0; w < 11; w++)
      words[w] = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, trailer + w * 4),
                               .align_mul = 4, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
   nir_def *indirect = nir_vec2(b, words[6], words[7]);
   nir_def *direct_grid = nir_vec3(b, words[3], words[4], words[5]);
   nir_push_if(b, nir_ine_imm(b, nir_ior(b, words[6], words[7]), 0));
   nir_def *loaded[3];
   for (unsigned axis = 0; axis < 3; axis++)
      loaded[axis] = nir_load_global_2x32(b, 1, 32,
         nir_build_addr_iadd_imm(b, indirect, nir_address_format_2x32bit_global,
                                 nir_var_mem_global, axis * 4), .align_mul = 4);
   nir_def *indirect_grid = nir_vec(b, loaded, 3);
   nir_def *indirect_end = nir_imul(b, nir_imul(b, loaded[0], loaded[1]), loaded[2]);
   nir_pop_if(b, NULL);
   nir_def *grid = nir_if_phi(b, indirect_grid, direct_grid);
   nir_def *end = nir_if_phi(b, indirect_end, words[1]);
   nir_store_var(b, ctx.grid, grid, 0x7);
   nir_variable *linear = nir_local_variable_create(impl, glsl_uint_type(), "linear");
   nir_def *native = nir_load_base_workgroup_id(b, 32);
   ctx.native = nir_def_as_intrinsic(native);
   nir_store_var(b, linear, nir_iadd(b, words[0], nir_channel(b, native, 0)), 1);
   nir_loop *loop = nir_push_loop(b);
   nir_def *l = nir_load_var(b, linear);
   nir_push_if(b, nir_uge(b, l, end));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_def *gx = nir_channel(b, grid, 0), *gy = nir_channel(b, grid, 1);
   nir_def *row = nir_udiv(b, l, gx);
   nir_def *origin = nir_vec3(b, words[8], words[9], words[10]);
   nir_store_var(b, ctx.id, nir_iadd(b, origin, nir_vec3(b, nir_isub(b, l, nir_imul(b, row, gx)),
                                                        nir_umod(b, row, gy), nir_udiv(b, row, gy))), 0x7);
   nir_cf_reinsert(&body, b->cursor);
   b->cursor = nir_after_cf_list(&loop->body);
   /* Shared memory of one workgroup is reused by the next. */
   if (nir->info.shared_size)
      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   nir_store_var(b, linear, nir_iadd(b, l, words[2]), 1);
   nir_pop_loop(b, loop);
   nir_progress(true, impl, nir_metadata_none);
   nir_shader_intrinsics_pass(nir, lower_workgroup, nir_metadata_control_flow, &ctx);
}

static bool
lower_push_constant(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   if (i->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   unsigned base = nir_intrinsic_base(i), size = ctx->program->push_size;
   bool aligned = (nir_intrinsic_align_mul(i) >= 4 && !(nir_intrinsic_align_offset(i) % 4)) ||
      (nir_src_is_const(i->src[0]) && !(nir_src_as_uint(i->src[0]) % 4));
   unsigned words = i->def.bit_size / 32;
   if ((i->def.bit_size != 32 && i->def.bit_size != 64) || i->num_components > 4 || !aligned ||
       base % 4 || base > size) {
      ctx->invalid = true;
      return false;
   }
   unsigned table_bytes = (ctx->program->descriptor_count + 1) * sizeof(union apex_descriptor);
   b->cursor = nir_before_instr(&i->instr);
   nir_def *values[8];
   for (unsigned c = 0; c < i->num_components * words; c++) {
      unsigned end = base + c * 4 + 4;
      if (end > size) {
         values[c] = nir_imm_int(b, 0);
         continue;
      }
      /* Clamping keeps the immutable load inside the push range, so it
       * speculates and hoists; words beyond the range read as zero. */
      nir_def *offset = i->src[0].ssa;
      nir_def *value = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
         nir_iadd_imm(b, nir_umin(b, offset, nir_imm_int(b, size - end)), table_bytes + base + c * 4),
         .align_mul = 4, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
      values[c] = nir_bcsel(b, nir_ule_imm(b, offset, size - end), value, nir_imm_int(b, 0));
   }
   if (words == 2)
      for (unsigned c = 0; c < i->num_components; c++)
         values[c] = nir_pack_64_2x32(b, nir_vec2(b, values[c * 2], values[c * 2 + 1]));
   nir_def_rewrite_uses(&i->def, nir_vec(b, values, i->num_components));
   nir_instr_remove(&i->instr);
   return true;
}

bool
apex_program_layout(struct apex_program *program, uint32_t set_count,
                    struct vk_descriptor_set_layout *const *sets,
                    uint32_t push_range_count, const VkPushConstantRange *ranges,
                    VkShaderStageFlags stages)
{
   program->table = true;
   program->set_count = set_count;
   for (unsigned s = 0; s < set_count; s++) {
      const struct apex_set_layout *set = (const void *)sets[s];
      program->set_layouts[s] = set;
      program->set_offsets[s] = program->descriptor_count;
      if (set) program->descriptor_count += set->slot_count;
   }
   if (program->descriptor_count > APEX_MAX_DESCRIPTORS)
      return false;
   for (unsigned r = 0; r < push_range_count; r++) {
      const VkPushConstantRange *range = &ranges[r];
      if (!(range->stageFlags & stages))
         continue;
      if (!range->size ||
          range->offset % 4 || range->size % 4 || range->offset >= APEX_MAX_PUSH_CONSTANTS ||
          range->size > APEX_MAX_PUSH_CONSTANTS - range->offset)
         return false;
      program->push_size = MAX2(program->push_size, range->offset + range->size);
   }
   return true;
}

bool
apex_program_lower_resources(struct apex_program *program, nir_shader *nir)
{
   const nir_lower_tex_options tex = {.lower_txp = ~0u, .lower_tg4_offsets = true};
   NIR_PASS(_, nir, nir_lower_tex, &tex);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);
   struct descriptor_lowering ctx = {.program = program};
   nir_shader_intrinsics_pass(nir, lower_resource, nir_metadata_control_flow, &ctx);
   nir_shader_intrinsics_pass(nir, lower_buffer, nir_metadata_none, &ctx);
   nir_shader_intrinsics_pass(nir, lower_image, nir_metadata_none, &ctx);
   nir_shader_intrinsics_pass(nir, lower_push_constant, nir_metadata_none, &ctx);
   /* Texture lowering reads table rows directly, so it follows buffer lowering. */
   if (ctx.invalid || !apex_lower_textures(program, nir))
      return false;
   NIR_PASS(_, nir, nir_lower_system_values);
   return true;
}

VkResult
apex_program_compile(struct vk_device *device, struct apex_program *program, nir_shader *nir)
{
   uint64_t invocations = nir->info.workgroup_size[0] * nir->info.workgroup_size[1] *
      nir->info.workgroup_size[2];
   if (apex_from_nir(nir, &program->code))
      return vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apex compile: %s", program->code.diagnostic);
   uint32_t private_bytes, instructions, vector;
   memcpy(&private_bytes, program->code.data + 28, sizeof(private_bytes));
   memcpy(&instructions, program->code.data + 8, sizeof(instructions));
   memcpy(&vector, program->code.data + 20, sizeof(vector));
   mesa_logd("Apex program: %u instructions, v%u, %u private bytes/lane, %u invocations",
             instructions, vector, private_bytes, (unsigned)invocations);
   const char *dump = os_get_option("APEX_DUMP_PROGRAMS");
   if (dump) {
      static unsigned serial;
      char path[512];
      snprintf(path, sizeof(path), "%s/program-%u.apx", dump, p_atomic_inc_return(&serial));
      FILE *f = fopen(path, "wb");
      if (f) {
         fwrite(program->code.data, 1, program->code.size, f);
         fclose(f);
      }
   }
   uint64_t extent = util_le32_to_cpu(private_bytes) * align64(invocations, 16);
   /* GPUVM reserves the low 2 MiB for this dispatch's padded private data. */
   program->max_workgroups = program->table && extent ?
      MIN2(1024, (2 * 1024 * 1024) / extent) : 1024;
   return program->max_workgroups ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

static const struct vk_pipeline_ops pipeline_ops = {
   .destroy = apex_pipeline_destroy,
};

static VkResult
create_compute_pipeline(struct vk_device *device,
                        const VkComputePipelineCreateInfo *info,
                        const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   VkPipelineCreateFlags2KHR flags = vk_compute_pipeline_create_flags(info);
   const VkPipelineCreateFlags2KHR supported =
      VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT |
      VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT |
      VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT |
      VK_PIPELINE_CREATE_2_ALLOW_DERIVATIVES_BIT | VK_PIPELINE_CREATE_2_DERIVATIVE_BIT |
      VK_PIPELINE_CREATE_2_DISPATCH_BASE_BIT;
   /* Subgroups are always full and 16 wide. */
   const VkPipelineShaderStageCreateFlags stage_flags =
      VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT |
      VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
   if ((flags & ~supported) || info->stage.stage != VK_SHADER_STAGE_COMPUTE_BIT ||
       (info->stage.flags & ~stage_flags))
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo *subgroup =
      vk_find_struct_const(info->stage.pNext,
                           PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
   if (subgroup && subgroup->requiredSubgroupSize != 16)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   /* No pipeline cache exists yet, so every accepted shader needs compilation. */
   if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
      return VK_PIPELINE_COMPILE_REQUIRED;

   const struct spirv_to_nir_options spirv_options = {
      .environment = NIR_SPIRV_VULKAN,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .skip_os_break_in_debug_build = true,
   };
   nir_shader *nir = NULL;
   VkResult result = vk_pipeline_shader_stage_to_nir(
      device, flags, &info->stage, &spirv_options, &apex_nir_options, NULL, &nir);
   if (result != VK_SUCCESS)
      return result;

   struct apex_pipeline *pipeline = vk_pipeline_zalloc(
      device, &pipeline_ops, VK_PIPELINE_BIND_POINT_COMPUTE, flags,
      alloc, sizeof(*pipeline));
   if (pipeline == NULL) {
      ralloc_free(nir);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   pipeline->vk.stages = VK_SHADER_STAGE_COMPUTE_BIT;
   struct vk_pipeline_layout *layout = vk_pipeline_layout_from_handle(info->layout);
   if (((struct apex_device *)device)->transport == APEX_TRANSPORT_DRM) {
      if (!layout)
         goto unsupported_layout;
      pipeline->layout = vk_pipeline_layout_ref(layout);
      if (!apex_program_layout(&pipeline->program, layout->set_count, layout->set_layouts,
                               layout->push_range_count, layout->push_ranges,
                               VK_SHADER_STAGE_COMPUTE_BIT))
         goto unsupported_layout;
      if (!apex_program_lower_resources(&pipeline->program, nir))
         goto unsupported_layout;
      wrap_workgroups(&pipeline->program, nir);
   }
   result = apex_program_compile(device, &pipeline->program, nir);
   ralloc_free(nir);
   if (result != VK_SUCCESS) {
      apex_pipeline_destroy(device, &pipeline->vk, alloc);
      return result;
   }
   *out = apex_pipeline_to_handle(pipeline);
   return VK_SUCCESS;

unsupported_layout:
   ralloc_free(nir);
   apex_pipeline_destroy(device, &pipeline->vk, alloc);
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateComputePipelines(VkDevice _device, VkPipelineCache cache,
                           uint32_t count,
                           const VkComputePipelineCreateInfo *infos,
                           const VkAllocationCallbacks *alloc,
                           VkPipeline *pipelines)
{
   VK_FROM_HANDLE(vk_device, device, _device);
   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < count; i++)
      pipelines[i] = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; i++) {
      VkResult r = create_compute_pipeline(device, &infos[i], alloc, &pipelines[i]);
      if (r == VK_SUCCESS)
         continue;
      /* A later compile-required status must not hide an earlier error. */
      if (result == VK_SUCCESS || r < 0)
         result = r;
      if (vk_compute_pipeline_create_flags(&infos[i]) &
          VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
         break;
   }
   return result;
}
