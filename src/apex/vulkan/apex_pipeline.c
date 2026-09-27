/* SPDX-License-Identifier: MIT */
#include "apex_pipeline.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_device.h"
#include "vk_log.h"

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
      if (layout->bindings[binding].type != type && layout->bindings[binding].type != dynamic_type) {
         ctx->invalid = true;
         return false;
      }
      nir_def *index = i->src[0].ssa;
      BITSET_SET_COUNT(ctx->program->used_descriptors,
         ctx->program->set_offsets[set] + layout->bindings[binding].offset,
         layout->bindings[binding].count);
      /* Opaque resource references retain the binding bounds through reindex.
       * The table is limited to 4096 entries, so base/count fit in 16 bits. */
      unsigned base = ctx->program->set_offsets[set] + layout->bindings[binding].offset;
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
   } else {
      nir_def *base = nir_vec2(b, words[0], words[1]);
      nir_def *offset = i->src[store ? 2 : 1].ssa;
      nir_def *values[4];
      for (unsigned c = 0; c < i->num_components; c++) {
         if (store && !(nir_intrinsic_write_mask(i) & (1u << c)))
            continue;
         /* Subtract from the range before comparing: offset + width may wrap. */
         nir_def *inside = nir_iand(b, nir_uge_imm(b, words[2], c * 4 + 4),
            nir_uge(b, nir_iadd_imm(b, words[2], -(int)(c * 4 + 4)), offset));
         nir_push_if(b, inside);
         nir_def *address = nir_build_addr_iadd(b, base, nir_address_format_2x32bit_global,
            nir_var_mem_global, nir_iadd_imm(b, offset, c * 4));
         nir_def *loaded = NULL;
         if (store)
            nir_store_global_2x32(b, nir_channel(b, i->src[0].ssa, c), address,
                                 .align_mul = 4, .access = nir_intrinsic_access(i));
         else if (swap)
            loaded = nir_global_atomic_swap_2x32(b, 32, address, i->src[2].ssa, i->src[3].ssa,
               .atomic_op = nir_intrinsic_atomic_op(i), .access = nir_intrinsic_access(i));
         else if (atomic)
            loaded = nir_global_atomic_2x32(b, 32, address, i->src[2].ssa,
               .atomic_op = nir_intrinsic_atomic_op(i), .access = nir_intrinsic_access(i));
         else
            loaded = nir_load_global_2x32(b, 1, 32, address,
                                        .align_mul = 4, .access = nir_intrinsic_access(i));
         nir_push_else(b, NULL);
         nir_def *zero = nir_imm_int(b, 0);
         nir_pop_if(b, NULL);
         if (!store)
            values[c] = nir_if_phi(b, loaded, zero);
      }
      if (!store)
         nir_def_rewrite_uses(&i->def, nir_vec(b, values, i->num_components));
   }
   nir_instr_remove(&i->instr);
   return true;
}

static bool
lower_image(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   if (i->intrinsic == nir_intrinsic_barrier) {
      nir_variable_mode modes = nir_intrinsic_memory_modes(i);
      if (!(modes & nir_var_image))
         return false;
      nir_intrinsic_set_memory_modes(i, (modes & ~nir_var_image) | nir_var_mem_global);
      return true;
   }
   bool load = i->intrinsic == nir_intrinsic_image_deref_load;
   bool store = i->intrinsic == nir_intrinsic_image_deref_store;
   bool swap = i->intrinsic == nir_intrinsic_image_deref_atomic_swap;
   bool atomic = swap || i->intrinsic == nir_intrinsic_image_deref_atomic;
   bool size = i->intrinsic == nir_intrinsic_image_deref_size;
   if (!load && !store && !atomic && !size)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(i->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   bool rgba = var && var->data.image.format == PIPE_FORMAT_R8G8B8A8_UNORM;
   if (!var || var->data.descriptor_set >= ctx->program->set_count ||
       nir_intrinsic_image_dim(i) != GLSL_SAMPLER_DIM_2D ||
       (var->data.image.format != PIPE_FORMAT_R32_UINT && !rgba) ||
       (rgba && atomic) ||
       (!store && i->def.bit_size != 32) ||
       (load && i->def.num_components != 4) ||
       (atomic && i->def.num_components != 1) ||
       (size && i->def.num_components != (nir_intrinsic_image_array(i) ? 3 : 2)) ||
       (store && (i->src[3].ssa->bit_size != 32 || i->src[3].ssa->num_components != 4))) {
      ctx->invalid = true;
      return false;
   }
   if (load || store || size) {
      nir_src lod = i->src[store ? 4 : size ? 1 : 3];
      if (!nir_src_is_const(lod) || nir_src_as_uint(lod)) {
         ctx->invalid = true;
         return false;
      }
   }
   unsigned set = var->data.descriptor_set, binding = var->data.binding;
   const struct apex_set_layout *layout = ctx->program->set_layouts[set];
   if (!layout || binding >= layout->binding_count || !layout->bindings[binding].count ||
       layout->bindings[binding].type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
      ctx->invalid = true;
      return false;
   }
   b->cursor = nir_before_instr(&i->instr);
   nir_def *index = nir_imm_int(b, 0);
   if (deref->deref_type == nir_deref_type_array) {
      index = deref->arr.index.ssa;
      deref = nir_deref_instr_parent(deref);
   }
   if (deref->deref_type != nir_deref_type_var) {
      ctx->invalid = true;
      return false;
   }
   unsigned base = ctx->program->set_offsets[set] + layout->bindings[binding].offset;
   BITSET_SET_COUNT(ctx->program->used_descriptors, base, layout->bindings[binding].count);
   nir_def *slot = nir_bcsel(b, nir_ult_imm(b, index, layout->bindings[binding].count),
      nir_iadd_imm(b, index, base), nir_imm_int(b, ctx->program->descriptor_count));
   nir_def *row = nir_imul_imm(b, slot, sizeof(union apex_descriptor));
   nir_def *words[7];
   for (unsigned c = 0; c < ARRAY_SIZE(words); c++)
      words[c] = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
         nir_iadd_imm(b, row, c * 4), .align_mul = 4,
         .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   if (size) {
      nir_def_rewrite_uses(&i->def, nir_vec(b, &words[2], i->def.num_components));
   } else {
      unsigned access = nir_intrinsic_access(i) | var->data.access;
      nir_def *coord = i->src[1].ssa;
      nir_def *x = nir_channel(b, coord, 0), *y = nir_channel(b, coord, 1);
      nir_def *inside = nir_iand(b, nir_ult(b, x, words[2]), nir_ult(b, y, words[3]));
      nir_def *offset = nir_iadd(b, nir_imul_imm(b, x, 4), nir_imul(b, y, words[5]));
      if (nir_intrinsic_image_array(i)) {
         nir_def *layer = nir_channel(b, coord, 2);
         inside = nir_iand(b, inside, nir_ult(b, layer, words[4]));
         offset = nir_iadd(b, offset, nir_imul(b, layer, words[6]));
      }
      nir_push_if(b, inside);
      nir_def *address = nir_build_addr_iadd(b, nir_vec2(b, words[0], words[1]),
         nir_address_format_2x32bit_global, nir_var_mem_global, offset);
      nir_def *value = NULL;
      if (store) {
         nir_def *pixel = nir_channel(b, i->src[3].ssa, 0);
         if (rgba) {
            pixel = nir_imm_int(b, 0);
            for (unsigned c = 0; c < 4; c++) {
               nir_def *channel = nir_channel(b, i->src[3].ssa, c);
               channel = nir_fmin(b, nir_fmax(b, channel, nir_imm_float(b, 0.0f)),
                                  nir_imm_float(b, 1.0f));
               nir_def *scaled = nir_fmul_imm(b, channel, 255.0f);
               nir_def *integer = nir_f2u32(b, scaled);
               nir_def *fraction = nir_fsub(b, scaled, nir_u2f32(b, integer));
               nir_def *round_up = nir_ior(b, nir_flt(b, nir_imm_float(b, 0.5f), fraction),
                  nir_iand(b, nir_feq_imm(b, fraction, 0.5f),
                           nir_ine_imm(b, nir_iand_imm(b, integer, 1), 0)));
               pixel = nir_ior(b, pixel,
                  nir_ishl_imm(b, nir_iadd(b, integer, nir_b2i32(b, round_up)), c * 8));
            }
         }
         nir_store_global_2x32(b, pixel, address, .align_mul = 4, .access = access);
      }
      else if (swap)
         value = nir_global_atomic_swap_2x32(b, 32, address, i->src[3].ssa, i->src[4].ssa,
            .atomic_op = nir_intrinsic_atomic_op(i), .access = access);
      else if (atomic)
         value = nir_global_atomic_2x32(b, 32, address, i->src[3].ssa,
            .atomic_op = nir_intrinsic_atomic_op(i), .access = access);
      else {
         value = nir_load_global_2x32(b, 1, 32, address,
                                    .align_mul = 4, .access = access);
         if (rgba) {
            nir_def *channels[4];
            for (unsigned c = 0; c < 4; c++)
               channels[c] = nir_fmul_imm(b,
                  nir_u2f32(b, nir_iand_imm(b, nir_ushr_imm(b, value, c * 8), 0xff)),
                  1.0f / 255.0f);
            value = nir_vec(b, channels, 4);
         } else {
            value = nir_vec4(b, value, nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 1));
         }
      }
      nir_push_else(b, NULL);
      nir_def *zero = nir_imm_zero(b, store ? 1 : i->def.num_components, 32);
      nir_pop_if(b, NULL);
      if (!store)
         nir_def_rewrite_uses(&i->def, nir_if_phi(b, value, zero));
   }
   nir_instr_remove(&i->instr);
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

static bool
lower_dispatch(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   bool count = i->intrinsic == nir_intrinsic_load_num_workgroups;
   bool global = i->intrinsic == nir_intrinsic_load_global_invocation_id;
   if (!count && !global && i->intrinsic != nir_intrinsic_load_workgroup_id)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   unsigned offset = apex_program_trailer(ctx->program);
   offset += count ? offsetof(struct apex_dispatch_parameters, groups) :
                     offsetof(struct apex_dispatch_parameters, base);
   nir_def *words[3];
   for (unsigned axis = 0; axis < 3; axis++)
      words[axis] = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
         nir_imm_int(b, offset + axis * 4), .align_mul = 4,
         .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   nir_def *replacement = nir_vec(b, words, 3);
   /* NIR's WorkgroupID system-value lowering already adds the native base
    * to load_workgroup_id. GlobalInvocationID needs that addition here. */
   if (global) {
      replacement = nir_iadd(b, replacement, nir_load_base_workgroup_id(b, 32));
      nir_def *size = nir_imm_ivec3(b, b->shader->info.workgroup_size[0],
         b->shader->info.workgroup_size[1], b->shader->info.workgroup_size[2]);
      replacement = nir_iadd(b, nir_imul(b, replacement, size),
                            nir_load_local_invocation_id(b));
   }
   nir_def_rewrite_uses(&i->def, replacement);
   nir_instr_remove(&i->instr);
   return true;
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
      nir_def *inside = end <= size ? nir_ule_imm(b, i->src[0].ssa, size - end) : nir_imm_false(b);
      nir_push_if(b, inside);
      nir_def *value = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
         nir_iadd_imm(b, i->src[0].ssa, table_bytes + base + c * 4), .align_mul = 4,
         .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_push_else(b, NULL);
      nir_def *zero = nir_imm_int(b, 0);
      nir_pop_if(b, NULL);
      values[c] = nir_if_phi(b, value, zero);
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
      if (set) program->descriptor_count += set->descriptor_count;
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
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);
   struct descriptor_lowering ctx = {.program = program};
   nir_shader_intrinsics_pass(nir, lower_resource, nir_metadata_control_flow, &ctx);
   nir_shader_intrinsics_pass(nir, lower_buffer, nir_metadata_none, &ctx);
   nir_shader_intrinsics_pass(nir, lower_image, nir_metadata_none, &ctx);
   nir_shader_intrinsics_pass(nir, lower_push_constant, nir_metadata_none, &ctx);
   NIR_PASS(_, nir, nir_lower_system_values);
   return !ctx.invalid;
}

VkResult
apex_program_compile(struct vk_device *device, struct apex_program *program, nir_shader *nir)
{
   uint64_t invocations = nir->info.workgroup_size[0] * nir->info.workgroup_size[1] *
      nir->info.workgroup_size[2];
   if (apex_from_nir(nir, &program->code))
      return vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apex compile: %s", program->code.diagnostic);
   uint32_t private_bytes;
   memcpy(&private_bytes, program->code.data + 28, sizeof(private_bytes));
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
      VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT;
   if ((flags & ~supported) || info->stage.stage != VK_SHADER_STAGE_COMPUTE_BIT ||
       info->stage.flags)
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
      struct descriptor_lowering ctx = {.program = &pipeline->program};
      if (!apex_program_lower_resources(&pipeline->program, nir))
         goto unsupported_layout;
      nir_shader_intrinsics_pass(nir, lower_dispatch, nir_metadata_control_flow, &ctx);
      if (ctx.invalid)
         goto unsupported_layout;
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
