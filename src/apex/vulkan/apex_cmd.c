/* SPDX-License-Identifier: MIT */
/* Command buffers as command-processor packets (Docs/architecture.md,
 * Command processor and user-mode rings; Fixed-function graphics). Each
 * command buffer is one IB. Dispatches and draws reference descriptor
 * tables, state blocks and programs whose GPUVAs are patched into the IB
 * copy at submission (apex_cmd_prepare). */
#include "apex_private.h"
#include "apex_format.h"
#include "apex_pipeline.h"
#include "vk_alloc.h"
#include "vk_buffer_view.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_update_template.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_image.h"
#include "vk_log.h"
#include "vk_meta.h"
#include "vk_pipeline_layout.h"
#include "vk_render_pass.h"
#include "util/format/u_format.h"
#include "util/rounding.h"
#include "util/u_math.h"

#define LO(v) ((uint32_t)(v))
#define HI(v) ((uint32_t)((uint64_t)(v) >> 32))

static uint32_t
float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

/* ---- Command buffer lifetime ------------------------------------------- */

static void
bound_set_unref(struct apex_command_buffer *cmd, struct apex_bound_set *bound)
{
   if (bound && !--bound->refs)
      vk_free(&cmd->vk.pool->alloc, bound);
}

static void
table_free(struct apex_command_buffer *cmd, struct apex_table *table)
{
   list_del(&table->link);
   for (unsigned s = 0; s < ARRAY_SIZE(table->sets); s++)
      bound_set_unref(cmd, table->sets[s]);
   vk_free(&cmd->vk.pool->alloc, table);
}

static void
clear_commands(struct apex_command_buffer *cmd)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   list_for_each_entry_safe(struct apex_upload, upload, &cmd->uploads, link) {
      list_del(&upload->link);
      if (upload->reserved_va) {
         mtx_lock(&device->va_mutex);
         util_vma_heap_free(&device->va_heap, upload->reserved_va, align64(upload->size, 4096));
         mtx_unlock(&device->va_mutex);
      }
      apex_bo_finish(device, &upload->bo);
      vk_free(&cmd->vk.pool->alloc, upload);
   }
   list_for_each_entry_safe(struct apex_descriptor_set, set, &cmd->push_sets, link)
      apex_descriptor_set_free(cmd->vk.base.device, set);
   memset(cmd->pushed, 0, sizeof(cmd->pushed));
   list_for_each_entry_safe(struct apex_table, table, &cmd->tables, link)
      table_free(cmd, table);
   for (unsigned s = 0; s < ARRAY_SIZE(cmd->sets); s++) {
      bound_set_unref(cmd, cmd->sets[s]);
      bound_set_unref(cmd, cmd->graphics_sets[s]);
   }
   memset(cmd->sets, 0, sizeof(cmd->sets));
   memset(cmd->graphics_sets, 0, sizeof(cmd->graphics_sets));
   util_dynarray_clear(&cmd->data);
   util_dynarray_clear(&cmd->patches);
   util_dynarray_clear(&cmd->pass_availability);
   cmd->pipeline = NULL;
   cmd->vertex = cmd->geometry = cmd->fragment = NULL;
   cmd->graphics_table = NULL;
   cmd->scratch = 0;
   cmd->private_pending = false;
   memset(cmd->bindings, 0, sizeof(cmd->bindings));
   memset(&cmd->index, 0, sizeof(cmd->index));
   memset(&cmd->state, 0, sizeof(cmd->state));
   memset(&cmd->xfb, 0, sizeof(cmd->xfb));
   cmd->query_slots = 0;
   cmd->meta = 0;
   memset(&cmd->predicate, 0, sizeof(cmd->predicate));
   memset(&cmd->rendering, 0, sizeof(cmd->rendering));
   memset(cmd->push, 0, sizeof(cmd->push));
   memset(cmd->graphics_push, 0, sizeof(cmd->graphics_push));
   apex_ib_reset(&cmd->ib);
   cmd->pending = 0;
}

static void
reset_command_buffer(struct vk_command_buffer *vk, VkCommandBufferResetFlags flags)
{
   clear_commands((struct apex_command_buffer *)vk);
   vk_command_buffer_reset(vk);
}

static void
destroy_command_buffer(struct vk_command_buffer *vk)
{
   struct apex_command_buffer *cmd = (void *)vk;
   struct vk_command_pool *pool = vk->pool;
   clear_commands(cmd);
   apex_ib_finish(&cmd->ib);
   util_dynarray_fini(&cmd->data);
   util_dynarray_fini(&cmd->patches);
   util_dynarray_fini(&cmd->pass_availability);
   vk_command_buffer_finish(vk);
   vk_free(&pool->alloc, vk);
}

static VkResult
create_command_buffer(struct vk_command_pool *pool, VkCommandBufferLevel level,
                      struct vk_command_buffer **out)
{
   struct apex_command_buffer *cmd = vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = vk_command_buffer_init_with_params(&cmd->vk,
      &(struct vk_command_buffer_init_params) {
         .pool = pool, .ops = &apex_command_buffer_ops, .level = level,
         .needs_cmd_queue = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY,
      });
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }
   list_inithead(&cmd->tables);
   list_inithead(&cmd->uploads);
   list_inithead(&cmd->push_sets);
   util_dynarray_init(&cmd->data, NULL);
   util_dynarray_init(&cmd->patches, NULL);
   util_dynarray_init(&cmd->pass_availability, NULL);
   apex_ib_init(&cmd->ib);
   cmd->vk.dynamic_graphics_state.vi = &cmd->vertex_input;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->sample_locations;
   *out = &cmd->vk;
   return VK_SUCCESS;
}

const struct vk_command_buffer_ops apex_command_buffer_ops = {
   .create = create_command_buffer, .reset = reset_command_buffer, .destroy = destroy_command_buffer,
};

/* Conditional rendering: the application's draws and dispatches and the
 * clears of vkCmdClearAttachments run under PREDICATE; driver work (vk_meta
 * transfers, internal kernels, render-pass clears) runs with it off. The CP
 * reads the predicate again each time PREDICATE turns it on. */
static bool
user_work(const struct apex_command_buffer *cmd)
{
   return !cmd->meta || cmd->predicate.meta;
}

static void
predicate(struct apex_command_buffer *cmd, bool user)
{
   bool on = user && cmd->predicate.active;
   if (on == cmd->predicate.on)
      return;
   apex_cp_predicate(&cmd->ib, on ? cmd->predicate.va : 0,
                     on ? APEX_CP_PREDICATE_ENABLE |
                          (cmd->predicate.inverted ? APEX_CP_PREDICATE_INVERTED : 0) : 0);
   cmd->predicate.on = on;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_BeginCommandBuffer(VkCommandBuffer handle, const VkCommandBufferBeginInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   vk_command_buffer_begin(&cmd->vk, info);
   /* The IB orders itself after earlier batches' work and host writes. */
   apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_ALL, APEX_CP_CACHE_L1 | APEX_CP_CACHE_TEXTURE);
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_EndCommandBuffer(VkCommandBuffer handle)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   /* The queue's predicate does not outlive the command buffer. */
   predicate(cmd, false);
   if (cmd->ib.failed || cmd->data.size == UINT32_MAX)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
   else if (cmd->ib.count > 1u << 24)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return vk_command_buffer_end(&cmd->vk);
}

/* ---- Submission data ---------------------------------------------------- */

static void
fail(struct apex_command_buffer *cmd, VkResult result)
{
   vk_command_buffer_set_error(&cmd->vk, result);
}

/* Appends a 64-byte aligned block to the command buffer's data. */
static uint64_t
data_block(struct apex_command_buffer *cmd, const void *bytes, unsigned size)
{
   uint64_t offset = align64(cmd->data.size, 64), pad = offset - cmd->data.size;
   uint8_t *out = util_dynarray_grow_bytes(&cmd->data, 1, pad + size);
   if (!out) {
      fail(cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
      return 0;
   }
   memset(out, 0, pad);
   memcpy(out + pad, bytes, size);
   return offset;
}

/* Reserves `size` bytes of scratch, 256-byte aligned: arena memory the
 * submission does not initialize. */
static uint64_t
scratch_block(struct apex_command_buffer *cmd, uint64_t size)
{
   uint64_t offset = align64(cmd->scratch, 256);
   cmd->scratch = offset + size;
   return offset;
}

static void
add_patch(struct apex_command_buffer *cmd, uint32_t dword, struct apex_patch patch)
{
   patch.dword = dword;
   util_dynarray_append(&cmd->patches, patch);
}

/* A table of `programs` over the given sets and push image. */
static struct apex_table *
table_create(struct apex_command_buffer *cmd, struct apex_program *a, struct apex_program *b,
             struct apex_bound_set *const *sets, const uint8_t *push)
{
   struct apex_table *table = vk_zalloc(&cmd->vk.pool->alloc, sizeof(*table), 8,
                                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!table) {
      fail(cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
      return NULL;
   }
   table->programs[0] = a;
   table->programs[1] = b;
   if (sets) {
      memcpy(table->sets, sets, sizeof(table->sets));
      for (unsigned s = 0; s < ARRAY_SIZE(table->sets); s++)
         if (table->sets[s])
            table->sets[s]->refs++;
   }
   if (push)
      memcpy(table->push, push, sizeof(table->push));
   list_addtail(&table->link, &cmd->tables);
   return table;
}

/* Work is tracked by class; BARRIER packets wait for what is pending. */
static void
begin_work(struct apex_command_buffer *cmd, uint32_t class)
{
   cmd->pending |= class;
}

/* ---- Compute ------------------------------------------------------------ */

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindPipeline(VkCommandBuffer handle, VkPipelineBindPoint point, VkPipeline pipeline)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      vk_common_CmdBindPipeline(handle, point, pipeline);
      return;
   }
   if (point != VK_PIPELINE_BIND_POINT_COMPUTE) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cmd->pipeline = apex_pipeline_from_handle(pipeline);
}

/* Compute and graphics bind points keep separate set bindings. */
static void
bind_set(struct apex_command_buffer *cmd, VkShaderStageFlags stages, uint32_t index,
         struct apex_bound_set *bound)
{
   if (stages & VK_SHADER_STAGE_COMPUTE_BIT) {
      bound_set_unref(cmd, cmd->sets[index]);
      cmd->sets[index] = bound;
   }
   if (stages & VK_SHADER_STAGE_ALL_GRAPHICS) {
      if (stages & VK_SHADER_STAGE_COMPUTE_BIT)
         bound->refs++;
      bound_set_unref(cmd, cmd->graphics_sets[index]);
      cmd->graphics_sets[index] = bound;
      cmd->graphics_table = NULL;
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindDescriptorSets2(VkCommandBuffer handle, const VkBindDescriptorSetsInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   uint32_t first = info->firstSet, count = info->descriptorSetCount;
   const VkDescriptorSet *sets = info->pDescriptorSets;
   const uint32_t *offsets = info->pDynamicOffsets;
   if (!(info->stageFlags & (VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS)) ||
       first > MESA_VK_MAX_DESCRIPTOR_SETS || count > MESA_VK_MAX_DESCRIPTOR_SETS - first) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   unsigned required = 0;
   for (unsigned s = 0; s < count; s++)
      required += apex_descriptor_set_from_handle(sets[s])->layout->vk.dynamic_descriptor_count;
   if (required != info->dynamicOffsetCount) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   unsigned consumed = 0;
   for (unsigned s = 0; s < count; s++) {
      struct apex_descriptor_set *set = apex_descriptor_set_from_handle(sets[s]);
      struct apex_bound_set *bound = vk_zalloc(&cmd->vk.pool->alloc,
         sizeof(*bound) + set->layout->vk.dynamic_descriptor_count * sizeof(bound->offsets[0]), 8,
         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!bound) {
         fail(cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      bound->set = set;
      bound->refs = 1;
      /* Consume in set, binding-number, array-element order, including unused bindings. */
      unsigned dynamic = set->layout->vk.dynamic_descriptor_count;
      memcpy(bound->offsets, offsets + consumed, dynamic * sizeof(bound->offsets[0]));
      consumed += dynamic;
      bind_set(cmd, info->stageFlags, first + s, bound);
   }
}

/* A push descriptor set: a command-buffer-owned set that starts from the
 * previous push to the same set index and layout, then binds. */
static struct apex_descriptor_set *
push_set(struct apex_command_buffer *cmd, VkShaderStageFlags stages, VkPipelineLayout handle,
         uint32_t index)
{
   VK_FROM_HANDLE(vk_pipeline_layout, layout, handle);
   struct apex_set_layout *set_layout = index < layout->set_count ? (void *)layout->set_layouts[index] : NULL;
   if (!set_layout || index >= MESA_VK_MAX_DESCRIPTOR_SETS) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return NULL;
   }
   struct apex_descriptor_set *set = vk_object_zalloc(cmd->vk.base.device, NULL,
      sizeof(*set) + set_layout->row_count * sizeof(set->rows[0]), VK_OBJECT_TYPE_DESCRIPTOR_SET);
   struct apex_bound_set *bound = vk_zalloc(&cmd->vk.pool->alloc, sizeof(*bound) +
      set_layout->vk.dynamic_descriptor_count * sizeof(bound->offsets[0]), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!set || !bound) {
      vk_object_free(cmd->vk.base.device, NULL, set);
      vk_free(&cmd->vk.pool->alloc, bound);
      fail(cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
      return NULL;
   }
   set->layout = (void *)vk_descriptor_set_layout_ref(&set_layout->vk);
   apex_descriptor_set_init(set, 0);
   const struct apex_descriptor_set *previous = cmd->pushed[index];
   if (previous && previous->layout == set_layout)
      memcpy(set->rows, previous->rows, set_layout->row_count * sizeof(set->rows[0]));
   list_addtail(&set->link, &cmd->push_sets);
   cmd->pushed[index] = set;
   bound->set = set;
   bound->refs = 1;
   bind_set(cmd, stages, index, bound);
   return set;
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdPushDescriptorSet2(VkCommandBuffer handle, const VkPushDescriptorSetInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_descriptor_set *set = push_set(cmd, info->stageFlags, info->layout, info->set);
   for (uint32_t i = 0; set && i < info->descriptorWriteCount; i++)
      apex_descriptor_set_write(set, &info->pDescriptorWrites[i]);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdPushDescriptorSetWithTemplate2(VkCommandBuffer handle,
                                       const VkPushDescriptorSetWithTemplateInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(vk_descriptor_update_template, templ, info->descriptorUpdateTemplate);
   struct apex_descriptor_set *set = push_set(cmd, vk_shader_stages_from_bind_point(templ->bind_point),
                                              info->layout, info->set);
   if (set)
      apex_descriptor_set_write_template(set, templ, info->pData);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdPushConstants2(VkCommandBuffer handle, const VkPushConstantsInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   uint32_t offset = info->offset, size = info->size;
   if (offset % 4 || size % 4 ||
       !size || offset >= APEX_MAX_PUSH_CONSTANTS || size > APEX_MAX_PUSH_CONSTANTS - offset) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   /* Compute and graphics bind points keep separate push images. */
   if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      memcpy(cmd->push + offset, info->pValues, size);
   if (info->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      memcpy(cmd->graphics_push + offset, info->pValues, size);
      cmd->graphics_table = NULL;
   }
}

/* The local size a program header holds (Docs/isa.md, Program header). */
static void
program_local_size(const struct apex_program *program, uint32_t local[3])
{
   const uint8_t *h = program->code.data;
   local[0] = h[16] | h[17] << 8;
   local[1] = h[18] | h[19] << 8;
   local[2] = h[20] | h[21] << 8;
}

static bool
program_private(const struct apex_program *program)
{
   return program->max_workgroups != UINT32_MAX;
}

/* SET_STATE of the compute registers and user data (s0:s1 table, s2:s3
 * grid address), then DISPATCH or DISPATCH_INDIRECT. Dispatches overlap
 * unless one uses the private arena, which they take in turn. */
static void
emit_dispatch(struct apex_command_buffer *cmd, struct apex_program *program, struct apex_table *table,
              const uint32_t base[3], const uint32_t grid[3], uint64_t indirect)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   if (program_private(program)) {
      if (cmd->private_pending)
         apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_COMPUTE, 0);
      cmd->private_pending = true;
   }
   begin_work(cmd, APEX_CP_CLASS_COMPUTE);
   uint32_t local[3];
   program_local_size(program, local);
   /* A command buffer runs on the one queue of its pool's family. */
   const uint64_t private_base = device->queues[cmd->vk.pool->queue_family_index].private_arena.va;
   const uint32_t compute[10] = {
      0, 0, local[0], local[1], local[2], base[0], base[1], base[2],
      LO(private_base), HI(private_base),
   };
   add_patch(cmd, cmd->ib.count + 2, (struct apex_patch){.kind = APEX_PATCH_PROGRAM, .program = program});
   apex_cp_set_state(&cmd->ib, APEX_STATE_COMPUTE_PROGRAM, ARRAY_SIZE(compute), compute);
   const uint32_t user[4] = {0, 0, LO(indirect), HI(indirect)};
   add_patch(cmd, cmd->ib.count + 2, (struct apex_patch){.kind = APEX_PATCH_TABLE, .table = table});
   if (!indirect)
      add_patch(cmd, cmd->ib.count + 4, (struct apex_patch){.kind = APEX_PATCH_GRID, .table = table});
   apex_cp_set_state(&cmd->ib, APEX_STATE_COMPUTE_USER, ARRAY_SIZE(user), user);
   if (indirect)
      apex_cp_dispatch_indirect(&cmd->ib, indirect);
   else
      apex_cp_dispatch(&cmd->ib, grid[0], grid[1], grid[2]);
}

/* Splits a grid into DISPATCHes whose waves fit the private arena. */
static void
dispatch_grid(struct apex_command_buffer *cmd, struct apex_program *program, struct apex_table *table,
              const uint32_t origin[3], const uint32_t grid[3])
{
   table->trailer[0] = grid[0];
   table->trailer[1] = grid[1];
   table->trailer[2] = grid[2];
   uint32_t limit = program->max_workgroups;
   if ((uint64_t)grid[0] * grid[1] * grid[2] <= limit) {
      emit_dispatch(cmd, program, table, origin, grid, 0);
      return;
   }
   uint32_t cx = MIN2(grid[0], limit), cy = MAX2(MIN2(grid[1], limit / cx), 1);
   for (uint32_t z = 0; z < grid[2]; z++) {
      for (uint32_t y = 0; y < grid[1]; y += cy) {
         for (uint32_t x = 0; x < grid[0]; x += cx) {
            const uint32_t base[3] = {origin[0] + x, origin[1] + y, origin[2] + z};
            const uint32_t chunk[3] = {MIN2(cx, grid[0] - x), MIN2(cy, grid[1] - y), 1};
            emit_dispatch(cmd, program, table, base, chunk, 0);
         }
      }
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDispatchBase(VkCommandBuffer handle, uint32_t ox, uint32_t oy, uint32_t oz,
                     uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (!x || !y || !z)
      return;
   if (!cmd->pipeline || !cmd->pipeline->program.table) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   predicate(cmd, user_work(cmd));
   struct apex_table *table = table_create(cmd, &cmd->pipeline->program, NULL, cmd->sets, cmd->push);
   if (table)
      dispatch_grid(cmd, &cmd->pipeline->program, table, (uint32_t[3]){ox, oy, oz}, (uint32_t[3]){x, y, z});
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDispatch(VkCommandBuffer handle, uint32_t x, uint32_t y, uint32_t z)
{
   apex_CmdDispatchBase(handle, 0, 0, 0, x, y, z);
}

/* The CP reads the grid when the packet executes; the program reads the
 * same VkDispatchIndirectCommand for its workgroup count. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdDispatchIndirect(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, args, buffer);
   if (!cmd->pipeline || !cmd->pipeline->program.table) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   predicate(cmd, user_work(cmd));
   struct apex_table *table = table_create(cmd, &cmd->pipeline->program, NULL, cmd->sets, cmd->push);
   if (table)
      emit_dispatch(cmd, &cmd->pipeline->program, table, (uint32_t[3]){0}, NULL,
                    args->vk.device_address + offset);
}

/* An internal kernel over a grid of workgroups with its job words; job
 * word w holds a data offset for bit w of `relocs`. `user` kernels belong
 * to an application draw and share its predicate. */
static void
internal_dispatch(struct apex_command_buffer *cmd, enum apex_internal which, const uint32_t *words,
                  const uint32_t grid[3], uint32_t relocs, bool user)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   if (!grid[0] || !grid[1] || !grid[2])
      return;
   predicate(cmd, user && user_work(cmd));
   struct apex_program *program;
   VkResult result = apex_internal_program(device, which, &program);
   if (result != VK_SUCCESS) {
      fail(cmd, result);
      return;
   }
   struct apex_table *table = table_create(cmd, program, NULL, NULL, NULL);
   if (!table)
      return;
   memcpy(&table->trailer[APEX_TRAILER_WORDS], words, APEX_JOB_WORDS * 4);
   table->relocs = (uint64_t)relocs << APEX_TRAILER_WORDS;
   dispatch_grid(cmd, program, table, (uint32_t[3]){0}, grid);
}

void
apex_cmd_bind_shaders(struct vk_command_buffer *vk, uint32_t count,
                      const mesa_shader_stage *stages, struct vk_shader **const shaders)
{
   struct apex_command_buffer *cmd = (void *)vk;
   for (uint32_t i = 0; i < count; i++) {
      struct apex_shader *shader = shaders[i] ? container_of(shaders[i], struct apex_shader, vk) : NULL;
      if (stages[i] == MESA_SHADER_VERTEX)
         cmd->vertex = shader;
      else if (stages[i] == MESA_SHADER_GEOMETRY)
         cmd->geometry = shader;
      else if (stages[i] == MESA_SHADER_FRAGMENT)
         cmd->fragment = shader;
   }
   cmd->graphics_table = NULL;
}

/* ---- vk_meta state ------------------------------------------------------ */

/* vk_meta binds its own pipelines, sets, push constants and dynamic state;
 * the application's return afterwards. */
struct saved_state {
   struct vk_dynamic_graphics_state dyn;
   struct vk_vertex_input_state vertex_input;
   struct vk_sample_locations_state sample_locations;
   struct apex_shader *vertex, *geometry, *fragment;
   struct apex_pipeline *pipeline;
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS], *graphics_sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   struct apex_descriptor_set *pushed[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint8_t push[APEX_MAX_PUSH_CONSTANTS], graphics_push[APEX_MAX_PUSH_CONSTANTS];
   struct apex_hw_binding bindings[APEX_HW_MAX_BINDINGS];
   uint16_t strides[MESA_VK_MAX_VERTEX_BINDINGS];
};

static void
meta_begin(struct apex_command_buffer *cmd, struct saved_state *s)
{
   s->dyn = cmd->vk.dynamic_graphics_state;
   s->vertex_input = cmd->vertex_input;
   s->sample_locations = cmd->sample_locations;
   s->vertex = cmd->vertex;
   s->geometry = cmd->geometry;
   s->fragment = cmd->fragment;
   s->pipeline = cmd->pipeline;
   memcpy(s->sets, cmd->sets, sizeof(s->sets));
   memcpy(s->graphics_sets, cmd->graphics_sets, sizeof(s->graphics_sets));
   for (unsigned i = 0; i < ARRAY_SIZE(s->sets); i++) {
      if (s->sets[i])
         s->sets[i]->refs++;
      if (s->graphics_sets[i])
         s->graphics_sets[i]->refs++;
   }
   memcpy(s->pushed, cmd->pushed, sizeof(s->pushed));
   memcpy(s->push, cmd->push, sizeof(s->push));
   memcpy(s->graphics_push, cmd->graphics_push, sizeof(s->graphics_push));
   memcpy(s->bindings, cmd->bindings, sizeof(s->bindings));
   cmd->meta++;
}

static void
meta_end(struct apex_command_buffer *cmd, struct saved_state *s)
{
   cmd->meta--;
   cmd->vk.dynamic_graphics_state = s->dyn;
   cmd->vertex_input = s->vertex_input;
   cmd->sample_locations = s->sample_locations;
   cmd->vk.dynamic_graphics_state.vi = &cmd->vertex_input;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->sample_locations;
   BITSET_ONES(cmd->vk.dynamic_graphics_state.dirty);
   cmd->vertex = s->vertex;
   cmd->geometry = s->geometry;
   cmd->fragment = s->fragment;
   cmd->pipeline = s->pipeline;
   for (unsigned i = 0; i < ARRAY_SIZE(s->sets); i++) {
      bound_set_unref(cmd, cmd->sets[i]);
      bound_set_unref(cmd, cmd->graphics_sets[i]);
   }
   memcpy(cmd->sets, s->sets, sizeof(s->sets));
   memcpy(cmd->graphics_sets, s->graphics_sets, sizeof(s->graphics_sets));
   memcpy(cmd->pushed, s->pushed, sizeof(s->pushed));
   memcpy(cmd->push, s->push, sizeof(s->push));
   memcpy(cmd->graphics_push, s->graphics_push, sizeof(s->graphics_push));
   memcpy(cmd->bindings, s->bindings, sizeof(s->bindings));
   cmd->graphics_table = NULL;
}

/* ---- Render passes ------------------------------------------------------ */

static struct apex_attachment
attachment(const VkRenderingAttachmentInfo *info, VkImageView resolve)
{
   VK_FROM_HANDLE(vk_image_view, view, resolve ? resolve : info ? info->imageView : VK_NULL_HANDLE);
   if (!view)
      return (struct apex_attachment){0};
   return (struct apex_attachment) {
      .image = (struct apex_image *)view->image, .format = view->format,
      .level = view->base_mip_level, .layer = view->base_array_layer, .aspects = view->aspects,
   };
}

/* An attachment's level at its first layer as the ROP addresses it. A 2D
 * view of a 3D image takes its slices as layers. */
static struct apex_hw_surface
surface_of(const struct apex_attachment *a)
{
   const struct apex_hw_layout *layout = &a->image->layout;
   const struct apex_hw_level *level = &layout->level[a->level];
   bool three_d = a->image->vk.image_type == VK_IMAGE_TYPE_3D;
   uint64_t layer_stride = three_d ? level->plane : layout->layer_stride;
   return (struct apex_hw_surface) {
      .va = apex_image_va(a->image) + level->offset + a->layer * layer_stride,
      .tiled = layout->tiled,
      .pitch = layout->tiled ? level->pitch : level->pitch / 64,
      .layer_stride = layer_stride / 64,
      .sample_stride = level->plane / 64,
   };
}

/* Tile-buffer planes of an attachment format (Docs/architecture.md, ROP
 * and tile buffer): 16-byte texels take two, others one. */
static unsigned
format_planes(VkFormat format)
{
   return vk_format_get_blocksize(format) > 8 ? 2 : 1;
}

static void
pack_color(VkFormat format, const VkClearColorValue *color, uint32_t texel[4])
{
   memset(texel, 0, 4 * sizeof(uint32_t));
   util_format_pack_rgba(vk_format_to_pipe_format(format), texel, color, 1);
}

/* The stored clear texel of a depth/stencil format, depth and stencil
 * packed in their memory positions (D32F_S8: depth dword 0, stencil 1). */
static void
pack_depth_stencil(VkFormat format, float depth, uint8_t stencil, uint32_t texel[2])
{
   texel[0] = texel[1] = 0;
   switch (format) {
   case VK_FORMAT_D16_UNORM: texel[0] = _mesa_lroundevenf(CLAMP(depth, 0, 1) * 65535.0f); break;
   case VK_FORMAT_X8_D24_UNORM_PACK32:
   case VK_FORMAT_D24_UNORM_S8_UINT:
      texel[0] = _mesa_lroundevenf(CLAMP(depth, 0, 1) * 16777215.0f) | (uint32_t)stencil << 24;
      break;
   case VK_FORMAT_D32_SFLOAT: texel[0] = float_bits(depth); break;
   case VK_FORMAT_S8_UINT: texel[0] = stencil; break;
   case VK_FORMAT_D32_SFLOAT_S8_UINT: texel[0] = float_bits(depth), texel[1] = stencil; break;
   default: break;
   }
}

static enum apex_hw_load
load_of(VkAttachmentLoadOp op)
{
   /* LOAD_OP_NONE loads: a pass split into raster phases stores every
    * attachment between them, which must leave an untouched one intact. */
   return op == VK_ATTACHMENT_LOAD_OP_LOAD || op == VK_ATTACHMENT_LOAD_OP_NONE ? APEX_HW_LOAD :
          op == VK_ATTACHMENT_LOAD_OP_CLEAR ? APEX_HW_CLEAR : APEX_HW_LOAD_DONT_CARE;
}

/* Region `index` of the pass's render area and the region count: bands of
 * whole tile rows, or runs of tiles along one row when a row alone takes
 * more bins than the raster back end. */
static unsigned
pass_region(const struct apex_hw_pass *pass, unsigned index, struct apex_hw_pass *region)
{
   *region = *pass;
   int selector = apex_hw_pass_selector(pass);
   if (selector < 0 || pass->x1 <= pass->x0 || pass->y1 <= pass->y0)
      return 1;
   unsigned tw = apex_hw_selector_width(selector), th = apex_hw_selector_height(selector);
   unsigned columns = (pass->x1 - 1) / tw - pass->x0 / tw + 1, rows = (pass->y1 - 1) / th - pass->y0 / th + 1;
   unsigned tiles = MAX2(APEX_HW_MAX_BINS / MAX2(pass->layers, 1), 1);
   if ((uint64_t)columns * rows <= tiles)
      return 1;
   unsigned band = columns <= tiles ? tiles / columns : 1, run = columns <= tiles ? columns : tiles;
   unsigned per_row = DIV_ROUND_UP(columns, run), count = DIV_ROUND_UP(rows, band) * per_row;
   unsigned row = index / per_row * band, column = index % per_row * run;
   unsigned tx0 = pass->x0 / tw + column, ty0 = pass->y0 / th + row;
   region->x0 = MAX2(pass->x0, tx0 * tw);
   region->y0 = MAX2(pass->y0, ty0 * th);
   region->x1 = MIN2(pass->x1, (tx0 + run) * tw);
   region->y1 = MIN2(pass->y1, (ty0 + band) * th);
   return count;
}

/* SET_STATE of a pass record, then BEGIN_PASS. Every pass bins into the
 * queue's one bin pool, so the pass waits for the raster phases of the
 * passes before it. */
static bool
begin_region(struct apex_command_buffer *cmd, unsigned index)
{
   struct apex_hw_pass region;
   pass_region(&cmd->rendering.pass, index, &region);
   uint32_t record[APEX_HW_PASS_DWORDS];
   if (!apex_hw_pass_record(&region, record)) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_GRAPHICS, 0);
   if (index == 0)
      cmd->rendering.record = cmd->ib.count + 2;
   apex_cp_set_state(&cmd->ib, APEX_STATE_PASS, APEX_HW_PASS_DWORDS, record);
   apex_cp_begin_pass(&cmd->ib);
   begin_work(cmd, APEX_CP_CLASS_GRAPHICS);
   return true;
}

/* Begins a pass in its first region. With more regions, the body's first
 * draw emits every state group, so each replay starts from it. */
static void
emit_pass(struct apex_command_buffer *cmd)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_hw_pass *pass = &cmd->rendering.pass, region;
   pass->pool = device->queues[0].bin_pool.va;
   pass->pool_bytes = APEX_BIN_POOL_BYTES;
   pass->draw_bytes = APEX_POOL_DRAW_BYTES;
   pass->vertex_bytes = APEX_POOL_VERTEX_BYTES;
   pass->primitive_bytes = APEX_POOL_PRIMITIVE_BYTES;
   cmd->rendering.regions = pass_region(pass, 0, &region);
   if (!begin_region(cmd, 0))
      return;
   cmd->rendering.active = true;
   if (cmd->rendering.regions > 1) {
      cmd->state.emitted = false;
      cmd->rendering.body = cmd->ib.count;
      cmd->rendering.body_predicate_va = cmd->predicate.on ? cmd->predicate.va : 0;
      cmd->rendering.body_predicate_flags = cmd->predicate.on ? APEX_CP_PREDICATE_ENABLE |
         (cmd->predicate.inverted ? APEX_CP_PREDICATE_INVERTED : 0) : 0;
   }
}

/* END_PASS, each further region replaying the body, then the availability
 * of the queries the pass ended. */
static void
end_pass(struct apex_command_buffer *cmd)
{
   uint32_t body = cmd->rendering.body, end = cmd->ib.count;
   apex_cp_end_pass(&cmd->ib);
   for (unsigned r = 1; r < cmd->rendering.regions; r++) {
      if (!begin_region(cmd, r))
         break;
      if (end > body) {
         apex_cp_predicate(&cmd->ib, cmd->rendering.body_predicate_va, cmd->rendering.body_predicate_flags);
         add_patch(cmd, cmd->ib.count + 1, (struct apex_patch){.kind = APEX_PATCH_IB, .offset = body});
         apex_cp_indirect(&cmd->ib, 0, end - body);
      }
      apex_cp_end_pass(&cmd->ib);
   }
   cmd->rendering.active = false;
   util_dynarray_foreach(&cmd->pass_availability, uint64_t, va)
      apex_cp_write(&cmd->ib, *va, APEX_CP_AFTER_PRIOR_WORK, 1, (uint32_t[1]){1});
   util_dynarray_clear(&cmd->pass_availability);
}

/* A barrier inside the pass (dynamic rendering local read) ends it with
 * every attachment stored and no resolve, rewriting the first region's
 * record, then begins it again loading them, so input attachments read the
 * writes before the barrier through the texture unit. Resolves wait for the
 * final end. */
static void
split_pass(struct apex_command_buffer *cmd)
{
   struct apex_hw_pass pass = cmd->rendering.pass, region;
   for (unsigned k = 0; k < ARRAY_SIZE(pass.attachment); k++) {
      cmd->rendering.pass.attachment[k].store |= pass.attachment[k].present;
      cmd->rendering.pass.attachment[k].resolve = false;
   }
   pass_region(&cmd->rendering.pass, 0, &region);
   if (!cmd->ib.failed)
      apex_hw_pass_record(&region, cmd->ib.words + cmd->rendering.record);
   end_pass(cmd);
   apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_GRAPHICS, APEX_CP_CACHE_L1 | APEX_CP_CACHE_TEXTURE);
   for (unsigned k = 0; k < ARRAY_SIZE(pass.attachment); k++)
      if (pass.attachment[k].present)
         pass.attachment[k].load = APEX_HW_LOAD;
   cmd->rendering.pass = pass;
   cmd->state.emitted = false;
   emit_pass(cmd);
}

static void
meta_rendering_info(const struct apex_command_buffer *cmd, struct vk_meta_rendering_info *info)
{
   *info = (struct vk_meta_rendering_info) {
      .view_mask = cmd->rendering.view_mask, .samples = cmd->rendering.samples,
      .color_attachment_count = cmd->rendering.color_count,
   };
   for (unsigned k = 0; k < cmd->rendering.color_count; k++) {
      info->color_attachment_formats[k] = cmd->rendering.color[k].format;
      info->color_attachment_write_masks[k] = 0xf;
   }
   if (cmd->rendering.has_depth)
      info->depth_attachment_format = cmd->rendering.depth.format;
   if (cmd->rendering.has_stencil)
      info->stencil_attachment_format = cmd->rendering.depth.format;
}

/* The application's clears are rendering commands under conditional
 * rendering; a render pass's own clears are not. */
static void
clear_attachments(struct apex_command_buffer *cmd, uint32_t count, const VkClearAttachment *attachments,
                  uint32_t rect_count, const VkClearRect *rects, bool user)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct vk_meta_rendering_info render;
   meta_rendering_info(cmd, &render);
   struct saved_state saved;
   meta_begin(cmd, &saved);
   cmd->predicate.meta = user;
   vk_meta_clear_attachments(&cmd->vk, &device->meta, &render, count, attachments, rect_count, rects);
   cmd->predicate.meta = false;
   meta_end(cmd, &saved);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdClearAttachments(VkCommandBuffer handle, uint32_t count, const VkClearAttachment *attachments,
                         uint32_t rect_count, const VkClearRect *rects)
{
   clear_attachments(apex_command_buffer_from_handle(handle), count, attachments, rect_count, rects, true);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBeginRendering(VkCommandBuffer handle, const VkRenderingInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_attachment depth = attachment(info->pDepthAttachment, VK_NULL_HANDLE);
   struct apex_attachment stencil = attachment(info->pStencilAttachment, VK_NULL_HANDLE);
   if (info->colorAttachmentCount > APEX_HW_MAX_COLOR || cmd->rendering.active ||
       (depth.image && stencil.image && (depth.image != stencil.image || depth.level != stencil.level ||
                                         depth.layer != stencil.layer))) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   memset(&cmd->rendering, 0, sizeof(cmd->rendering));
   cmd->rendering.area = info->renderArea;
   cmd->rendering.view_mask = info->viewMask;
   cmd->rendering.layers = info->viewMask ? util_last_bit(info->viewMask) : MAX2(info->layerCount, 1);
   cmd->rendering.color_count = info->colorAttachmentCount;
   cmd->rendering.suspending = info->flags & VK_RENDERING_SUSPENDING_BIT;
   /* A resumed pass continues the suspended one's attachments. */
   bool resuming = info->flags & VK_RENDERING_RESUMING_BIT;
   struct apex_hw_pass *pass = &cmd->rendering.pass;
   memset(pass, 0, sizeof(*pass));
   int64_t x1 = (int64_t)info->renderArea.offset.x + info->renderArea.extent.width;
   int64_t y1 = (int64_t)info->renderArea.offset.y + info->renderArea.extent.height;
   pass->x0 = MAX2(info->renderArea.offset.x, 0);
   pass->y0 = MAX2(info->renderArea.offset.y, 0);
   pass->x1 = CLAMP(x1, pass->x0, 0xffff);
   pass->y1 = CLAMP(y1, pass->y0, 0xffff);
   pass->layers = cmd->rendering.layers;
   unsigned samples = 1;
   for (unsigned k = 0; k < info->colorAttachmentCount; k++) {
      const VkRenderingAttachmentInfo *a = &info->pColorAttachments[k];
      struct apex_attachment color = attachment(a, VK_NULL_HANDLE);
      cmd->rendering.color[k] = color;
      if (!color.image)
         continue;
      samples = color.image->vk.samples;
      struct apex_hw_attachment *hw = &pass->attachment[k];
      hw->present = true;
      hw->format = apex_hw_color_format(color.format);
      hw->planes = format_planes(color.format);
      hw->load = resuming ? APEX_HW_LOAD : load_of(a->loadOp);
      hw->store = cmd->rendering.suspending || a->storeOp == VK_ATTACHMENT_STORE_OP_STORE;
      hw->surface = surface_of(&color);
      pack_color(color.format, &a->clearValue.color, hw->clear);
      struct apex_attachment target = attachment(NULL, a->resolveImageView);
      if (a->resolveMode != VK_RESOLVE_MODE_NONE && target.image && !cmd->rendering.suspending) {
         hw->resolve = true;
         hw->resolve_surface = surface_of(&target);
      }
   }
   /* One depth-stencil attachment: the depth view, else the stencil view.
    * Its aspects share one load op: an aspect that clears while the other
    * loads, or is not attached, loads and clears with a draw. */
   struct apex_attachment ds = depth.image ? depth : stencil;
   cmd->rendering.depth = ds;
   cmd->rendering.has_depth = depth.image;
   cmd->rendering.has_stencil = stencil.image;
   VkImageAspectFlags clear = 0;
   VkClearDepthStencilValue clear_value = {0};
   if (ds.image) {
      samples = ds.image->vk.samples;
      const VkRenderingAttachmentInfo *ops[2] = {depth.image ? info->pDepthAttachment : NULL,
                                                 stencil.image ? info->pStencilAttachment : NULL};
      const VkImageAspectFlags format_aspects = vk_format_aspects(ds.format);
      enum apex_hw_load load = APEX_HW_LOAD_DONT_CARE;
      bool store = cmd->rendering.suspending, all_clear = true, any_load = false;
      for (unsigned i = 0; i < 2; i++) {
         VkImageAspectFlags aspect = i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
         if (!(format_aspects & aspect))
            continue;
         if (!ops[i]) {
            /* An unattached aspect keeps its contents. */
            any_load = true, all_clear = false, store = true;
            continue;
         }
         VkAttachmentLoadOp op = resuming ? VK_ATTACHMENT_LOAD_OP_LOAD : ops[i]->loadOp;
         any_load |= op == VK_ATTACHMENT_LOAD_OP_LOAD || op == VK_ATTACHMENT_LOAD_OP_NONE;
         all_clear &= op == VK_ATTACHMENT_LOAD_OP_CLEAR;
         if (op == VK_ATTACHMENT_LOAD_OP_CLEAR) {
            clear |= aspect;
            if (i)
               clear_value.stencil = ops[i]->clearValue.depthStencil.stencil;
            else
               clear_value.depth = ops[i]->clearValue.depthStencil.depth;
         }
         store |= ops[i]->storeOp == VK_ATTACHMENT_STORE_OP_STORE;
      }
      load = all_clear ? APEX_HW_CLEAR : any_load || clear ? APEX_HW_LOAD : APEX_HW_LOAD_DONT_CARE;
      if (all_clear)
         clear = 0;
      struct apex_hw_attachment *hw = &pass->attachment[APEX_HW_DEPTH_ATTACHMENT];
      hw->present = true;
      hw->format = apex_hw_depth_format(ds.format);
      hw->planes = 1;
      hw->load = load;
      hw->store = store;
      hw->surface = surface_of(&ds);
      hw->clear_depth = clear_value.depth;
      const VkRenderingAttachmentInfo *op = ops[0] ? ops[0] : ops[1];
      pack_depth_stencil(ds.format, op->clearValue.depthStencil.depth,
                         ops[1] ? ops[1]->clearValue.depthStencil.stencil : 0, hw->clear);
      if (all_clear)
         hw->clear_depth = op->clearValue.depthStencil.depth;
      struct apex_attachment target = attachment(NULL, op->resolveImageView);
      if (op->resolveMode != VK_RESOLVE_MODE_NONE && target.image && !cmd->rendering.suspending) {
         hw->resolve = true;
         hw->resolve_surface = surface_of(&target);
      }
   }
   pass->samples = samples;
   cmd->rendering.samples = samples;
   emit_pass(cmd);
   if (clear) {
      const VkClearAttachment c = {.aspectMask = clear, .clearValue.depthStencil = clear_value};
      const VkClearRect rect = {.rect = info->renderArea, .layerCount = info->viewMask ? 1 : cmd->rendering.layers};
      clear_attachments(cmd, 1, &c, 1, &rect, false);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdEndRendering(VkCommandBuffer handle)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (cmd->rendering.active)
      end_pass(cmd);
}

/* ---- Draws -------------------------------------------------------------- */

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindVertexBuffers2(VkCommandBuffer handle, uint32_t first, uint32_t count,
                           const VkBuffer *buffers, const VkDeviceSize *offsets,
                           const VkDeviceSize *sizes, const VkDeviceSize *strides)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   for (uint32_t i = 0; i < count && first + i < APEX_HW_MAX_BINDINGS; i++) {
      VK_FROM_HANDLE(apex_buffer, buffer, buffers[i]);
      /* A null buffer binds zero bytes: fetches read zero. */
      cmd->bindings[first + i].va = buffer ? buffer->vk.device_address + offsets[i] : 0;
      cmd->bindings[first + i].size = !buffer ? 0 :
         sizes && sizes[i] != VK_WHOLE_SIZE ? sizes[i] : buffer->vk.size - offsets[i];
   }
   if (strides)
      vk_cmd_set_vertex_binding_strides(&cmd->vk, first, count, strides);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindIndexBuffer2(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                         VkDeviceSize size, VkIndexType type)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, index, buffer);
   /* Index types: 0 8-bit, 1 16-bit, 2 32-bit. */
   cmd->index.type = type == VK_INDEX_TYPE_UINT32 ? 2 : type == VK_INDEX_TYPE_UINT16 ? 1 : 0;
   cmd->index.va = index ? index->vk.device_address + offset : 0;
   cmd->index.size = !index ? 0 : size == VK_WHOLE_SIZE ? index->vk.size - offset : size;
}

/* A state block in the command buffer's data, reused while unchanged. */
static uint64_t
state_block(struct apex_command_buffer *cmd, unsigned which, uint32_t *last, const uint32_t *words,
            unsigned dwords)
{
   if (cmd->state.emitted && cmd->state.blocks[which] != UINT64_MAX && !memcmp(last, words, dwords * 4))
      return cmd->state.blocks[which];
   memcpy(last, words, dwords * 4);
   cmd->state.blocks[which] = data_block(cmd, words, dwords * 4);
   return cmd->state.blocks[which];
}

/* Writes `count` registers when they differ from the last emitted values;
 * `patches` names the dwords holding GPUVAs (two dwords each). */
static void
emit_registers(struct apex_command_buffer *cmd, uint32_t index, uint32_t *last, const uint32_t *values,
               unsigned count, const struct apex_patch *patches, unsigned patch_count, bool changed)
{
   if (cmd->state.emitted && !changed && !memcmp(last, values, count * 4))
      return;
   memcpy(last, values, count * 4);
   for (unsigned p = 0; p < patch_count; p++)
      add_patch(cmd, cmd->ib.count + 2 + patches[p].dword, patches[p]);
   apex_cp_set_state(&cmd->ib, index, count, values);
}

static uint32_t
hw_topology(VkPrimitiveTopology topology)
{
   /* vk_meta rectangles arrive as triangle lists. */
   return topology == VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA ? 3 : MIN2(topology, 5);
}

static uint64_t
primitive_count(VkPrimitiveTopology topology, uint64_t vertices)
{
   switch (hw_topology(topology)) {
   case 0: return vertices;
   case 1: return vertices / 2;
   case 2: return vertices > 1 ? vertices - 1 : 0;
   case 3: return vertices / 3;
   default: return vertices > 2 ? vertices - 2 : 0;
   }
}

/* The rasterized stage of a geometry shader's draw: the pass-through
 * program's table and the index list at scratch offset `list`, of list
 * topology `topology`. */
struct geometry_draw {
   struct apex_table *table;
   uint64_t list, list_bytes;
   VkPrimitiveTopology topology;
};

/* The draw state registers, blocks and tables, emitted where they changed;
 * false when the draw cannot run. */
static bool
emit_draw_state(struct apex_command_buffer *cmd, bool indexed, uint32_t view, const struct geometry_draw *geometry)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!cmd->rendering.active || !cmd->vertex ||
       (!geometry && dyn->ia.primitive_topology > VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
        dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA)) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   struct apex_program *vs = geometry ? &cmd->geometry->geometry.draw : &cmd->vertex->program, *fs;
   if (cmd->fragment) {
      fs = &cmd->fragment->program;
   } else {
      VkResult result = apex_empty_fragment_program(device, &fs);
      if (result != VK_SUCCESS) {
         fail(cmd, result);
         return false;
      }
   }
   if (geometry) {
      /* The pass-through program reads the records through its own table. */
      cmd->graphics_table = geometry->table;
      geometry->table->programs[1] = fs;
   } else if (!cmd->graphics_table) {
      cmd->graphics_table = table_create(cmd, vs, fs, cmd->graphics_sets, cmd->graphics_push);
      if (!cmd->graphics_table)
         return false;
   }
   if (!cmd->state.emitted)
      for (unsigned b = 0; b < ARRAY_SIZE(cmd->state.blocks); b++)
         cmd->state.blocks[b] = UINT64_MAX;
   /* Blocks: vertex input, raster, depth-stencil, blend, viewport. */
   uint32_t vi[APEX_HW_VERTEX_INPUT_DWORDS], raster[APEX_HW_RASTER_DWORDS];
   uint32_t ds[APEX_HW_DEPTH_STENCIL_DWORDS], blend[APEX_HW_BLEND_DWORDS], vp[APEX_HW_VIEWPORT_DWORDS];
   apex_hw_vertex_input_block(dyn, cmd->bindings, geometry ? cmd->geometry->geometry.clip_distances :
                              cmd->vertex->vertex.clip_distances, !geometry && cmd->vertex->vertex.robust, vi);
   if (geometry) {
      struct vk_dynamic_graphics_state list = *dyn;
      list.ia.primitive_topology = geometry->topology;
      apex_hw_raster_block(&list, apex_program_sample_shading(fs), apex_program_late_depth(fs), raster);
   } else {
      apex_hw_raster_block(dyn, apex_program_sample_shading(fs), apex_program_late_depth(fs), raster);
   }
   const struct apex_attachment *z = &cmd->rendering.depth;
   apex_hw_depth_stencil_block(dyn, z->image ? z->format : VK_FORMAT_UNDEFINED, cmd->rendering.has_depth,
                               cmd->rendering.has_stencil, ds);
   apex_hw_blend_block(dyn, cmd->rendering.color_count, blend);
   apex_hw_viewport_block(dyn, vp);
   uint64_t blocks[5] = {
      state_block(cmd, 0, cmd->state.vertex_input, vi, ARRAY_SIZE(vi)),
      state_block(cmd, 1, cmd->state.raster, raster, ARRAY_SIZE(raster)),
      state_block(cmd, 2, cmd->state.depth_stencil, ds, ARRAY_SIZE(ds)),
      state_block(cmd, 3, cmd->state.blend, blend, ARRAY_SIZE(blend)),
      state_block(cmd, 4, cmd->state.viewport, vp, ARRAY_SIZE(vp)),
   };
   /* Block identity is its data offset; the patch writes its GPUVA. */
   uint32_t vertex[11] = {
      0, 0, LO(blocks[0]), HI(blocks[0]),
      indexed ? LO(cmd->index.va) : 0, indexed ? HI(cmd->index.va) : 0,
      indexed ? (uint32_t)MIN2(cmd->index.size, UINT32_MAX) : 0, indexed ? cmd->index.type : 0,
      hw_topology(dyn->ia.primitive_topology), indexed && dyn->ia.primitive_restart_enable, ~0u,
   };
   const struct apex_patch vertex_patches[3] = {
      {.dword = 0, .kind = APEX_PATCH_PROGRAM, .program = vs},
      {.dword = 2, .kind = APEX_PATCH_DATA, .offset = blocks[0]},
      {.dword = 4, .kind = APEX_PATCH_SCRATCH, .offset = geometry ? geometry->list : 0},
   };
   if (geometry) {
      /* The 32-bit index list; the scratch patch writes its address. */
      vertex[4] = LO(geometry->list);
      vertex[5] = HI(geometry->list);
      vertex[6] = (uint32_t)MIN2(geometry->list_bytes, UINT32_MAX);
      vertex[7] = 2;
      vertex[8] = hw_topology(geometry->topology);
      vertex[9] = 0;
   }
   emit_registers(cmd, APEX_STATE_VERTEX, cmd->state.vertex, vertex, ARRAY_SIZE(vertex), vertex_patches,
                  geometry ? 3 : 2, cmd->state.vertex_program != vs);
   cmd->state.vertex_program = vs;
   /* Vertex user data: s0:s1 table, s2 view index. */
   if (!cmd->state.emitted || cmd->state.table != cmd->graphics_table || cmd->state.view != view) {
      const uint32_t user[3] = {0, 0, view};
      add_patch(cmd, cmd->ib.count + 2, (struct apex_patch){.kind = APEX_PATCH_TABLE,
                                                             .table = cmd->graphics_table});
      apex_cp_set_state(&cmd->ib, APEX_STATE_VERTEX_USER, ARRAY_SIZE(user), user);
   }
   const uint32_t fragment[10] = {
      0, 0, LO(blocks[1]), HI(blocks[1]), LO(blocks[2]), HI(blocks[2]),
      LO(blocks[3]), HI(blocks[3]), LO(blocks[4]), HI(blocks[4]),
   };
   const struct apex_patch fragment_patches[5] = {
      {.dword = 0, .kind = APEX_PATCH_PROGRAM, .program = fs},
      {.dword = 2, .kind = APEX_PATCH_DATA, .offset = blocks[1]},
      {.dword = 4, .kind = APEX_PATCH_DATA, .offset = blocks[2]},
      {.dword = 6, .kind = APEX_PATCH_DATA, .offset = blocks[3]},
      {.dword = 8, .kind = APEX_PATCH_DATA, .offset = blocks[4]},
   };
   emit_registers(cmd, APEX_STATE_FRAGMENT, cmd->state.fragment, fragment, ARRAY_SIZE(fragment),
                  fragment_patches, 5, cmd->state.fragment_program != fs);
   cmd->state.fragment_program = fs;
   /* Fragment user data: s0:s1 table. */
   if (!cmd->state.emitted || cmd->state.table != cmd->graphics_table) {
      add_patch(cmd, cmd->ib.count + 2, (struct apex_patch){.kind = APEX_PATCH_TABLE,
                                                             .table = cmd->graphics_table});
      apex_cp_set_state(&cmd->ib, APEX_STATE_FRAGMENT_USER, 2, (uint32_t[2]){0, 0});
   }
   cmd->state.table = cmd->graphics_table;
   cmd->state.view = view;
   uint32_t dynamic[APEX_HW_DYNAMIC_DWORDS];
   apex_hw_dynamic_registers(dyn, dynamic);
   emit_registers(cmd, APEX_STATE_DYNAMIC, cmd->state.dynamic, dynamic, ARRAY_SIZE(dynamic), NULL, 0, false);
   cmd->state.emitted = true;
   begin_work(cmd, APEX_CP_CLASS_GRAPHICS);
   return true;
}

/* Indirect draw source: the command address, its stride and draw count,
 * and the count buffer address (zero without one). */
struct indirect_draw {
   uint64_t va, count_va;
   uint32_t count, stride;
};

/* The job words describing draw `draw` (APEX_DRAW_*): its command words or
 * indirect command, count buffer, index buffer, flags with `topology`, and
 * the vertex input block at data offset `vertex_input`. */
static void
draw_words(struct apex_command_buffer *cmd, bool indexed, const uint32_t *packet,
           const struct indirect_draw *indirect, uint32_t draw, uint32_t topology, uint64_t vertex_input,
           uint32_t words[APEX_JOB_WORDS])
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   memset(words, 0, APEX_JOB_WORDS * 4);
   words[APEX_DRAW_NUMBER] = draw;
   words[APEX_DRAW_FLAGS] = (indexed ? cmd->index.type | APEX_DRAW_FLAG_INDEXED : 0) |
                            (dyn->ia.primitive_restart_enable ? APEX_DRAW_FLAG_RESTART : 0) |
                            (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ?
                             APEX_DRAW_FLAG_LAST : 0) | topology << 8;
   words[APEX_DRAW_VERTEX_INPUT] = LO(vertex_input);
   words[APEX_DRAW_VERTEX_INPUT + 1] = HI(vertex_input);
   if (indexed) {
      words[APEX_DRAW_INDICES] = LO(cmd->index.va);
      words[APEX_DRAW_INDICES + 1] = HI(cmd->index.va);
      words[APEX_DRAW_INDEX_BYTES] = MIN2(cmd->index.size, UINT32_MAX);
   }
   if (indirect) {
      uint64_t params = indirect->va + (uint64_t)draw * indirect->stride;
      words[APEX_DRAW_INDIRECT] = LO(params);
      words[APEX_DRAW_INDIRECT + 1] = HI(params);
      words[APEX_DRAW_COUNT] = LO(indirect->count_va);
      words[APEX_DRAW_COUNT + 1] = HI(indirect->count_va);
   } else {
      memcpy(&words[APEX_DRAW_COMMAND], packet, (indexed ? 5 : 4) * 4);
   }
}

/* Transform feedback and the stream query of one draw: after the compute
 * work before it, the capture program writes the captured outputs; after
 * that the bookkeeping kernel advances the offsets and the query. Driver
 * draws (vk_meta) are not counted. */
static void
record_xfb(struct apex_command_buffer *cmd, bool indexed, const uint32_t *packet,
           const struct indirect_draw *indirect, uint32_t draw)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct apex_program *capture = cmd->xfb.active && cmd->vertex->vertex.capture.code.size ?
      &cmd->vertex->vertex.capture : NULL;
   if (cmd->meta || (!capture && !cmd->xfb.query))
      return;
   uint32_t topology = hw_topology(dyn->ia.primitive_topology);
   uint32_t words[APEX_JOB_WORDS];
   draw_words(cmd, indexed, packet, indirect, draw, topology, cmd->state.blocks[0], words);
   words[APEX_XFB_QUERY] = LO(cmd->xfb.query);
   words[APEX_XFB_QUERY + 1] = HI(cmd->xfb.query);
   words[APEX_XFB_MODE] = APEX_XFB_ADVANCE;
   uint32_t relocs = BITFIELD_BIT(APEX_DRAW_VERTEX_INPUT);
   if (cmd->xfb.active) {
      words[APEX_XFB_STATE] = LO(cmd->xfb.state);
      words[APEX_XFB_STATE + 1] = HI(cmd->xfb.state);
      relocs |= BITFIELD_BIT(APEX_XFB_STATE);
   }
   for (unsigned b = 0; b < APEX_XFB_BUFFERS; b++)
      words[APEX_XFB_STRIDES + b / 2] |= (uint32_t)cmd->vertex->vertex.xfb_strides[b] << 16 * (b & 1);
   uint32_t groups = 64;
   if (!indirect) {
      uint64_t vertices = primitive_count(dyn->ia.primitive_topology, packet[0]) *
                          (topology == 0 ? 1 : topology <= 2 ? 2 : 3) * (uint64_t)packet[1];
      groups = MIN2(DIV_ROUND_UP(vertices, APEX_DRAW_LOCAL), 4096);
   }
   if (capture && groups) {
      struct apex_table *table = table_create(cmd, capture, NULL, cmd->graphics_sets, cmd->graphics_push);
      if (!table)
         return;
      groups = MIN2(groups, capture->max_workgroups);
      table->trailer[0] = groups;
      table->trailer[1] = table->trailer[2] = 1;
      memcpy(&table->trailer[APEX_TRAILER_WORDS], words, sizeof(words));
      table->relocs = (uint64_t)relocs << APEX_TRAILER_WORDS;
      apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_COMPUTE, APEX_CP_CACHE_L1);
      emit_dispatch(cmd, capture, table, (uint32_t[3]){0}, (uint32_t[3]){groups, 1, 1}, 0);
   }
   apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_COMPUTE, APEX_CP_CACHE_L1);
   internal_dispatch(cmd, APEX_INTERNAL_XFB, words, (uint32_t[3]){1, 1, 1}, relocs, true);
}

/* Input primitives of `vertices` positions, bounding a restarted strip's. */
static uint64_t
input_primitives(VkPrimitiveTopology topology, uint64_t vertices)
{
   static const unsigned size[10] = {1, 2, 2, 3, 3, 3, 4, 4, 6, 6};
   static const unsigned shared[10] = {0, 0, 1, 0, 2, 2, 0, 3, 0, 4};
   if (topology >= ARRAY_SIZE(size))
      return 0;
   if (!shared[topology])
      return vertices / size[topology];
   return vertices < size[topology] ? 0 : (vertices - shared[topology]) / (size[topology] - shared[topology]);
}

/* Scratch for one geometry draw. A direct draw takes what its invocations
 * need, an indirect draw a fixed budget; both stop at the budget, and
 * invocations beyond it draw nothing. */
#define APEX_GS_BUDGET (64ull * 1024 * 1024)

/* A draw through a geometry shader (apex_graphics.c, Geometry shading): the
 * records and geometry programs, then the index list's draw, split to fit
 * the bin pool's vertex region when direct. */
static void
record_geometry(struct apex_command_buffer *cmd, bool indexed, const uint32_t *packet,
                const struct indirect_draw *indirect, uint32_t draw, uint32_t view)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct apex_shader *gs = cmd->geometry;
   struct apex_program *records = &cmd->vertex->program, *geometry = &gs->program;
   const unsigned n = gs->geometry.primitive_vertices;
   const uint64_t in_stride = 16 * util_bitcount64(gs->geometry.inputs);
   const uint64_t out_stride = 16 * util_bitcount64(gs->geometry.outputs);
   const uint64_t list_bytes = 4ull * gs->geometry.max_primitives * n;
   const uint64_t per_invocation = gs->geometry.max_vertices * out_stride + list_bytes;
   uint64_t inputs = APEX_GS_BUDGET / 4 / in_stride, invocations = APEX_GS_BUDGET * 3 / 4 / per_invocation;
   if (!indirect) {
      inputs = MIN2(inputs, (uint64_t)packet[0] * packet[1]);
      invocations = MIN2(invocations, input_primitives(dyn->ia.primitive_topology, packet[0]) * packet[1] *
                                      gs->geometry.invocations);
   }
   if (!inputs || !invocations)
      return;
   /* The vertex shader's own vertex input block, with its clip distances
    * and robustness. */
   uint32_t vi[APEX_HW_VERTEX_INPUT_DWORDS];
   apex_hw_vertex_input_block(dyn, cmd->bindings, cmd->vertex->vertex.clip_distances, cmd->vertex->vertex.robust,
                              vi);
   uint64_t block = data_block(cmd, vi, sizeof(vi));
   uint32_t words[APEX_JOB_WORDS];
   draw_words(cmd, indexed, packet, indirect, draw, dyn->ia.primitive_topology, block, words);
   uint64_t record_va = scratch_block(cmd, inputs * in_stride);
   uint64_t output_va = scratch_block(cmd, invocations * gs->geometry.max_vertices * out_stride);
   uint64_t list_va = scratch_block(cmd, invocations * list_bytes);
   uint64_t arguments_va = indirect ? scratch_block(cmd, 32) : 0;
   const uint64_t scratch[4][2] = {
      {APEX_GS_RECORDS, record_va}, {APEX_GS_OUTPUT, output_va}, {APEX_GS_LIST, list_va},
      {APEX_GS_ARGUMENTS, arguments_va},
   };
   uint64_t scratch_relocs = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(scratch); i++) {
      if (i == 3 && !indirect)
         continue;
      words[scratch[i][0]] = LO(scratch[i][1]);
      words[scratch[i][0] + 1] = HI(scratch[i][1]);
      scratch_relocs |= BITFIELD64_BIT(scratch[i][0]);
   }
   words[APEX_GS_CAPACITY] = invocations;
   words[APEX_GS_RECORD_CAPACITY] = inputs;
   /* Records, then the geometry program, each after the work before it. */
   const uint64_t work[2] = {inputs, invocations};
   struct apex_program *programs[2] = {records, geometry};
   for (unsigned k = 0; k < 2; k++) {
      struct apex_table *table = table_create(cmd, programs[k], NULL, cmd->graphics_sets, cmd->graphics_push);
      if (!table)
         return;
      uint32_t groups = MIN3(DIV_ROUND_UP(work[k], APEX_DRAW_LOCAL), 4096, programs[k]->max_workgroups);
      table->trailer[0] = groups;
      table->trailer[1] = table->trailer[2] = 1;
      memcpy(&table->trailer[APEX_TRAILER_WORDS], words, sizeof(words));
      table->relocs = (uint64_t)BITFIELD_BIT(APEX_DRAW_VERTEX_INPUT) << APEX_TRAILER_WORDS;
      table->scratch_relocs = scratch_relocs << APEX_TRAILER_WORDS;
      apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_COMPUTE, APEX_CP_CACHE_L1);
      emit_dispatch(cmd, programs[k], table, (uint32_t[3]){0}, (uint32_t[3]){groups, 1, 1}, 0);
   }
   apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_COMPUTE, APEX_CP_CACHE_L1);
   /* The pass-through program's table: the output records and capacity. */
   struct apex_table *table = table_create(cmd, &gs->geometry.draw, NULL, cmd->graphics_sets, cmd->graphics_push);
   if (!table)
      return;
   memcpy(&table->trailer[APEX_TRAILER_WORDS], words, sizeof(words));
   table->scratch_relocs = scratch_relocs << APEX_TRAILER_WORDS;
   static const VkPrimitiveTopology lists[4] = {
      0, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const struct geometry_draw list = {
      .table = table, .list = list_va, .list_bytes = invocations * list_bytes, .topology = lists[n],
   };
   if (!emit_draw_state(cmd, true, view, &list))
      return;
   cmd->graphics_table = NULL;
   if (indirect) {
      add_patch(cmd, cmd->ib.count + 1, (struct apex_patch){.kind = APEX_PATCH_SCRATCH, .offset = arguments_va});
      apex_cp_draw_indirect(&cmd->ib, true, 0, 1, 20, 0);
      return;
   }
   /* Pieces of whole list primitives whose vertices fit the vertex region. */
   uint64_t indices = invocations * gs->geometry.max_primitives * n;
   uint32_t capacity = APEX_POOL_VERTEX_BYTES / MAX2(gs->geometry.stride, 64) - 4 * 32;
   capacity -= capacity % n;
   for (uint64_t first = 0; first < indices; first += capacity)
      apex_cp_draw_indexed(&cmd->ib, MIN2(capacity, indices - first), 1, first, 0, 0);
}

/* Multiview repeats the draw per view with the view index in user data;
 * the vertex program writes it as the layer. Transform feedback is not
 * active with multiview. A pass whose bin pool fills splits into raster
 * phases in the CP. */
static void
record_draw(struct apex_command_buffer *cmd, bool indexed, const uint32_t *packet,
            const struct indirect_draw *indirect)
{
   if (!cmd->rendering.active || !cmd->vertex) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   predicate(cmd, user_work(cmd));
   uint32_t views = cmd->rendering.view_mask ? cmd->rendering.view_mask : 1;
   if (cmd->geometry) {
      u_foreach_bit(view, views) {
         for (uint32_t d = 0; d < (indirect ? indirect->count : 1); d++)
            record_geometry(cmd, indexed, packet, indirect, d, cmd->rendering.view_mask ? view : 0);
      }
      return;
   }
   u_foreach_bit(view, views) {
      if (!emit_draw_state(cmd, indexed, cmd->rendering.view_mask ? view : 0, NULL))
         return;
      for (uint32_t d = 0; view == ffs(views) - 1 && d < (indirect ? indirect->count : 1); d++)
         record_xfb(cmd, indexed, packet, indirect, d);
      if (indirect)
         apex_cp_draw_indirect(&cmd->ib, indexed, indirect->va, indirect->count, indirect->stride,
                               indirect->count_va);
      else if (indexed)
         apex_cp_draw_indexed(&cmd->ib, packet[0], packet[1], packet[2], (int32_t)packet[3], packet[4]);
      else
         apex_cp_draw(&cmd->ib, packet[0], packet[1], packet[2], packet[3]);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer handle, uint32_t first, uint32_t count,
                                        const VkBuffer *buffers, const VkDeviceSize *offsets,
                                        const VkDeviceSize *sizes)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   for (uint32_t i = 0; i < count && first + i < APEX_XFB_BUFFERS; i++) {
      VK_FROM_HANDLE(apex_buffer, buffer, buffers[i]);
      cmd->xfb.buffers[first + i].va = buffer->vk.device_address + offsets[i];
      cmd->xfb.buffers[first + i].size = sizes && sizes[i] != VK_WHOLE_SIZE ? sizes[i] :
                                         buffer->vk.size - offsets[i];
   }
}

/* Loads or stores the offsets of the buffers that have counter buffers. */
static void
xfb_counters(struct apex_command_buffer *cmd, uint32_t mode, uint32_t first, uint32_t count,
             const VkBuffer *counters, const VkDeviceSize *offsets)
{
   uint32_t words[APEX_JOB_WORDS] = {
      [APEX_XFB_STATE] = LO(cmd->xfb.state), [APEX_XFB_STATE + 1] = HI(cmd->xfb.state),
      [APEX_XFB_MODE] = mode,
   };
   bool any = false;
   for (uint32_t i = 0; counters && i < count && first + i < APEX_XFB_BUFFERS; i++) {
      VK_FROM_HANDLE(apex_buffer, counter, counters[i]);
      if (!counter)
         continue;
      uint64_t va = counter->vk.device_address + (offsets ? offsets[i] : 0);
      words[APEX_XFB_COUNTERS + 2 * (first + i)] = LO(va);
      words[APEX_XFB_COUNTERS + 2 * (first + i) + 1] = HI(va);
      any = true;
   }
   if (!any)
      return;
   apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_COMPUTE, APEX_CP_CACHE_L1);
   internal_dispatch(cmd, APEX_INTERNAL_XFB, words, (uint32_t[3]){1, 1, 1}, BITFIELD_BIT(APEX_XFB_STATE), false);
}

/* A new state block: zero offsets, or the counter buffers' when resuming. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdBeginTransformFeedbackEXT(VkCommandBuffer handle, uint32_t first, uint32_t count,
                                  const VkBuffer *counters, const VkDeviceSize *offsets)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   uint32_t state[APEX_XFB_STATE_WORDS] = {0};
   for (unsigned b = 0; b < APEX_XFB_BUFFERS; b++) {
      uint32_t *w = &state[APEX_XFB_STATE_BUFFERS + 3 * b];
      w[0] = LO(cmd->xfb.buffers[b].va);
      w[1] = HI(cmd->xfb.buffers[b].va);
      w[2] = MIN2(cmd->xfb.buffers[b].size, UINT32_MAX);
   }
   cmd->xfb.state = data_block(cmd, state, sizeof(state));
   cmd->xfb.active = true;
   xfb_counters(cmd, APEX_XFB_LOAD, first, count, counters, offsets);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdEndTransformFeedbackEXT(VkCommandBuffer handle, uint32_t first, uint32_t count,
                                const VkBuffer *counters, const VkDeviceSize *offsets)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (cmd->xfb.active)
      xfb_counters(cmd, APEX_XFB_STORE, first, count, counters, offsets);
   cmd->xfb.active = false;
}

/* A resumed draw shades again from its first vertex, so every draw's vertex
 * outputs must fit the bin pool's vertex region. A larger direct draw splits
 * into draws that fit: lists at whole primitives, strips repeating their
 * shared vertices from an even start, instances apart. Fans, adjacency,
 * patches and restarted index strips draw whole. */
static void
record_direct(struct apex_command_buffer *cmd, bool indexed, const uint32_t *packet)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const unsigned instance = indexed ? 4 : 3;
   uint32_t piece[5];
   memcpy(piece, packet, (instance + 1) * sizeof(uint32_t));
   /* A pass whose vertex region fills within assembly's four wave banks of
    * its end stalls for up to seconds on every resume, or never finishes. */
   uint32_t capacity = cmd->vertex ? APEX_POOL_VERTEX_BYTES / MAX2(cmd->vertex->vertex.stride, 64) - 4 * 32 :
                       UINT32_MAX;
   unsigned align, overlap;
   switch (dyn->ia.primitive_topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: align = 1; overlap = 0; break;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: align = 2; overlap = 0; break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: case VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA:
      align = 3; overlap = 0; break;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: align = 1; overlap = 1; break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: align = 2; overlap = 2; break;
   default: align = 0; overlap = 0; break;
   }
   /* A geometry shader's draw splits its own index list instead. */
   if ((uint64_t)packet[0] * packet[1] <= capacity || !align || cmd->geometry ||
       (indexed && overlap && dyn->ia.primitive_restart_enable)) {
      record_draw(cmd, indexed, packet, NULL);
      return;
   }
   if (packet[0] <= capacity) {
      for (uint32_t i = 0, step = capacity / packet[0]; i < packet[1]; i += step) {
         piece[1] = MIN2(step, packet[1] - i);
         piece[instance] = packet[instance] + i;
         record_draw(cmd, indexed, piece, NULL);
      }
      return;
   }
   piece[1] = 1;
   for (uint32_t i = 0; i < packet[1]; i++) {
      piece[instance] = packet[instance] + i;
      for (uint32_t start = 0; start + overlap < packet[0];) {
         uint32_t n = MIN2(packet[0] - start, capacity);
         if (start + n < packet[0])
            n -= (n - overlap) % align;
         piece[0] = n;
         piece[2] = packet[2] + start;
         record_draw(cmd, indexed, piece, NULL);
         start += n - overlap;
      }
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDraw(VkCommandBuffer handle, uint32_t vertex_count, uint32_t instance_count,
             uint32_t first_vertex, uint32_t first_instance)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (vertex_count && instance_count)
      record_direct(cmd, false, (uint32_t[4]){vertex_count, instance_count, first_vertex, first_instance});
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDrawIndexed(VkCommandBuffer handle, uint32_t index_count, uint32_t instance_count,
                    uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (index_count && instance_count)
      record_direct(cmd, true, (uint32_t[5]){index_count, instance_count, first_index,
                                             (uint32_t)vertex_offset, first_instance});
}

static void
record_indirect(struct apex_command_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset,
                VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t draws, uint32_t stride)
{
   VK_FROM_HANDLE(apex_buffer, args, buffer);
   VK_FROM_HANDLE(apex_buffer, counts, count_buffer);
   if (!draws)
      return;
   const struct indirect_draw indirect = {
      .va = args->vk.device_address + offset, .count = draws,
      .stride = draws > 1 ? stride : indexed ? 20 : 16,
      .count_va = counts ? counts->vk.device_address + count_offset : 0,
   };
   record_draw(cmd, indexed, NULL, &indirect);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDrawIndirect(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset, uint32_t draws,
                     uint32_t stride)
{
   record_indirect(apex_command_buffer_from_handle(handle), false, buffer, offset, VK_NULL_HANDLE, 0,
                   draws, stride);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDrawIndexedIndirect(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset, uint32_t draws,
                            uint32_t stride)
{
   record_indirect(apex_command_buffer_from_handle(handle), true, buffer, offset, VK_NULL_HANDLE, 0,
                   draws, stride);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDrawIndirectCount(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                          VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t draws,
                          uint32_t stride)
{
   record_indirect(apex_command_buffer_from_handle(handle), false, buffer, offset, count_buffer,
                   count_offset, draws, stride);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDrawIndexedIndirectCount(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                                 VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t draws,
                                 uint32_t stride)
{
   record_indirect(apex_command_buffer_from_handle(handle), true, buffer, offset, count_buffer,
                   count_offset, draws, stride);
}

/* ---- Conditional rendering ---------------------------------------------- */

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBeginConditionalRenderingEXT(VkCommandBuffer handle, const VkConditionalRenderingBeginInfoEXT *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, buffer, info->buffer);
   cmd->predicate.va = buffer->vk.device_address + info->offset;
   cmd->predicate.inverted = info->flags & VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;
   cmd->predicate.active = true;
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdEndConditionalRenderingEXT(VkCommandBuffer handle)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   cmd->predicate.active = false;
   predicate(cmd, false);
}

/* ---- Events, queries and timestamps ------------------------------------- */

/* Writes one dword after the queue's earlier work. */
static void
record_word(struct apex_command_buffer *cmd, uint64_t va, uint32_t value)
{
   apex_cp_write(&cmd->ib, va, APEX_CP_AFTER_PRIOR_WORK, 1, &value);
}

/* Fills words in recording order through the copy engine. */
static void
record_fill(struct apex_command_buffer *cmd, uint64_t va, uint64_t size, uint32_t value)
{
   begin_work(cmd, APEX_CP_CLASS_COPY);
   apex_cp_fill(&cmd->ib, va, size, value);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdSetEvent2(VkCommandBuffer handle, VkEvent event, const VkDependencyInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   record_word(cmd, apex_event_from_handle(event)->bo.va, 1);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdResetEvent2(VkCommandBuffer handle, VkEvent event, VkPipelineStageFlags2 stage)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   record_word(cmd, apex_event_from_handle(event)->bo.va, 0);
}

/* The event's u64 (state word, zero word) is at least one once set. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdWaitEvents2(VkCommandBuffer handle, uint32_t count, const VkEvent *events,
                    const VkDependencyInfo *infos)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   for (uint32_t i = 0; i < count; i++)
      apex_cp_wait(&cmd->ib, apex_event_from_handle(events[i])->bo.va, 1);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdResetQueryPool(VkCommandBuffer handle, VkQueryPool pool, uint32_t first, uint32_t count)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (count)
      record_fill(cmd, apex_query_pool_from_handle(pool)->bo.va + (uint64_t)first * APEX_QUERY_STRIDE,
                  (uint64_t)count * APEX_QUERY_STRIDE, 0);
}

/* Availability after the query's results: after END_PASS inside a pass. */
static void
query_available(struct apex_command_buffer *cmd, uint64_t va)
{
   /* Under multiview a query takes one index per view: the first holds the
    * result and the others keep the zero of their reset. */
   unsigned views = MAX2(util_bitcount(cmd->rendering.view_mask), 1);
   for (unsigned v = 0; v < views; v++) {
      uint64_t available = va + (uint64_t)v * APEX_QUERY_STRIDE + APEX_QUERY_AVAILABLE;
      if (cmd->rendering.active)
         util_dynarray_append(&cmd->pass_availability, available);
      else
         record_word(cmd, available, 1);
   }
}

/* Occlusion queries take one of the raster back end's eight counter slots
 * from QUERY_BEGIN to QUERY_END, which writes the 64-bit count. Stream
 * queries start from zero and count in each draw's transform feedback
 * bookkeeping. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdBeginQueryIndexedEXT(VkCommandBuffer handle, VkQueryPool pool, uint32_t query,
                             VkQueryControlFlags flags, uint32_t index)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_query_pool *qp = apex_query_pool_from_handle(pool);
   uint64_t va = qp->bo.va + (uint64_t)query * APEX_QUERY_STRIDE;
   if (qp->vk.query_type == VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT) {
      apex_cp_write(&cmd->ib, va, APEX_CP_AFTER_PRIOR_WORK, 4, (uint32_t[4]){0});
      cmd->xfb.query = va;
      return;
   }
   if (cmd->query_slots == 0xff) {
      fail(cmd, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   unsigned slot = ffs(~cmd->query_slots) - 1;
   cmd->query_slots |= 1u << slot;
   for (unsigned q = 0; q < ARRAY_SIZE(cmd->queries); q++) {
      if (!cmd->queries[q].va) {
         cmd->queries[q].va = va;
         cmd->queries[q].slot = slot;
         break;
      }
   }
   apex_cp_query_begin(&cmd->ib, slot);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdEndQueryIndexedEXT(VkCommandBuffer handle, VkQueryPool pool, uint32_t query, uint32_t index)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_query_pool *qp = apex_query_pool_from_handle(pool);
   uint64_t va = qp->bo.va + (uint64_t)query * APEX_QUERY_STRIDE;
   if (qp->vk.query_type == VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT) {
      cmd->xfb.query = 0;
      query_available(cmd, va);
      return;
   }
   for (unsigned q = 0; q < ARRAY_SIZE(cmd->queries); q++) {
      if (cmd->queries[q].va != va)
         continue;
      unsigned slot = cmd->queries[q].slot;
      cmd->queries[q].va = 0;
      cmd->query_slots &= ~(1u << slot);
      apex_cp_query_end(&cmd->ib, slot, va);
      query_available(cmd, va);
      return;
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdWriteTimestamp2(VkCommandBuffer handle, VkPipelineStageFlags2 stage, VkQueryPool pool,
                        uint32_t query)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   uint64_t slot = apex_query_pool_from_handle(pool)->bo.va + (uint64_t)query * APEX_QUERY_STRIDE;
   /* Value, then availability; both after earlier work unless top of pipe. */
   apex_cp_timestamp(&cmd->ib, slot, stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT ||
                     stage == VK_PIPELINE_STAGE_2_NONE ? 0 : APEX_CP_AFTER_PRIOR_WORK);
   for (unsigned v = 0; v < MAX2(util_bitcount(cmd->rendering.view_mask), 1); v++)
      record_word(cmd, slot + (uint64_t)v * APEX_QUERY_STRIDE + APEX_QUERY_AVAILABLE, 1);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyQueryPoolResults(VkCommandBuffer handle, VkQueryPool pool, uint32_t first,
                             uint32_t count, VkBuffer buffer, VkDeviceSize offset,
                             VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, dst, buffer);
   struct apex_query_pool *qp = apex_query_pool_from_handle(pool);
   uint64_t slot = qp->bo.va + (uint64_t)first * APEX_QUERY_STRIDE;
   uint64_t va = dst->vk.device_address + offset;
   uint32_t words[APEX_JOB_WORDS] = {
      [APEX_QUERY_SLOT] = LO(slot), [APEX_QUERY_SLOT + 1] = HI(slot),
      [APEX_QUERY_DST] = LO(va), [APEX_QUERY_DST + 1] = HI(va),
      [APEX_QUERY_DST_STRIDE] = stride, [APEX_QUERY_COUNT] = count, [APEX_QUERY_FLAGS] = flags,
      [APEX_QUERY_VALUES] = qp->vk.query_type == VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT ? 2 : 1,
   };
   internal_dispatch(cmd, APEX_INTERNAL_QUERY_COPY, words, (uint32_t[3]){DIV_ROUND_UP(count, 32), 1, 1}, 0, false);
}

/* ---- Transfers ---------------------------------------------------------- */

static VKAPI_ATTR void VKAPI_CALL
apex_CmdFillBuffer(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                   VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, dst, buffer);
   VkDeviceAddressRangeKHR range = vk_device_address_range(&dst->vk, offset, size);
   /* VK_WHOLE_SIZE leaves the final incomplete word untouched. */
   range.size &= ~3ull;
   if (range.size)
      record_fill(cmd, range.address, range.size, data);
}

static bool
buffer_system(const struct apex_buffer *buffer)
{
   return buffer->memory && buffer->memory->storage->bo.system;
}

/* The CP copies LOCAL to LOCAL at any alignment and LOCAL to or from SYSTEM
 * with 64-byte aligned ends; other regions use vk_meta's compute copy. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyBuffer2(VkCommandBuffer handle, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, src, info->srcBuffer);
   VK_FROM_HANDLE(apex_buffer, dst, info->dstBuffer);
   struct apex_device *device = (void *)cmd->vk.base.device;
   bool src_system = buffer_system(src), dst_system = buffer_system(dst);
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferCopy2 *region = &info->pRegions[i];
      uint64_t from = src->vk.device_address + region->srcOffset;
      uint64_t to = dst->vk.device_address + region->dstOffset;
      bool aligned = !((from | to | region->size) & 63);
      if (!src_system && !dst_system ? true : src_system != dst_system && aligned) {
         begin_work(cmd, APEX_CP_CLASS_COPY);
         apex_cp_copy(&cmd->ib, from, to, region->size);
         continue;
      }
      VkCopyBufferInfo2 one = *info;
      one.regionCount = 1;
      one.pRegions = region;
      struct saved_state saved;
      meta_begin(cmd, &saved);
      vk_meta_copy_buffer(&cmd->vk, &device->meta, &one);
      meta_end(cmd, &saved);
   }
}

/* At most 65536 bytes, a multiple of four: one WRITE after earlier work. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdUpdateBuffer(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                     VkDeviceSize size, const void *data)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, dst, buffer);
   if (size)
      apex_cp_write(&cmd->ib, dst->vk.device_address + offset, APEX_CP_AFTER_PRIOR_WORK,
                    size / 4, data);
}

/* The kernels' surface words of one image level from (x, y) of `layer`.
 * Planes are layers times samples, or the 3D slices from `layer`. */
static void
image_surface(const struct apex_image *image, const struct apex_hw_layout *layout, uint64_t va,
              unsigned level, unsigned layer, unsigned x, unsigned y, uint32_t words[APEX_SURFACE_WORDS])
{
   const struct apex_hw_level *l = &layout->level[level];
   bool three_d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   uint64_t layer_stride = three_d ? l->plane : layout->layer_stride;
   uint64_t base = va + l->offset + layer * layer_stride;
   words[APEX_SURFACE_VA] = LO(base);
   words[APEX_SURFACE_VA + 1] = HI(base);
   words[APEX_SURFACE_FLAGS] = (layout->tiled ? APEX_SURFACE_TILED : 0) |
                               (layout->tiled ? util_logbase2(layout->bytes) << 1 : 0) |
                               (three_d ? 1 : layout->samples) << 4;
   words[APEX_SURFACE_PITCH] = l->pitch;
   words[APEX_SURFACE_LAYER] = layer_stride;
   words[APEX_SURFACE_SAMPLE] = l->plane;
   words[APEX_SURFACE_ORIGIN] = x | y << 16;
   words[APEX_SURFACE_TEXEL] = layout->bytes;
}

static void
buffer_surface(uint64_t va, uint32_t row, uint32_t slice, uint32_t texel, uint32_t words[APEX_SURFACE_WORDS])
{
   memset(words, 0, APEX_SURFACE_WORDS * 4);
   words[APEX_SURFACE_VA] = LO(va);
   words[APEX_SURFACE_VA + 1] = HI(va);
   words[APEX_SURFACE_FLAGS] = 1 << 4;
   words[APEX_SURFACE_PITCH] = row;
   words[APEX_SURFACE_LAYER] = slice;
   words[APEX_SURFACE_TEXEL] = texel;
}

/* One element copy between surfaces (apex_job.h, copy.comp). */
static void
copy_surfaces(struct apex_command_buffer *cmd, const uint32_t src[APEX_SURFACE_WORDS],
              const uint32_t dst[APEX_SURFACE_WORDS], uint32_t width, uint32_t height, uint32_t planes,
              uint32_t bytes, uint32_t src_offset, uint32_t dst_offset, uint32_t mask)
{
   uint32_t words[APEX_JOB_WORDS] = {0};
   memcpy(&words[APEX_COPY_SRC], src, APEX_SURFACE_WORDS * 4);
   memcpy(&words[APEX_COPY_DST], dst, APEX_SURFACE_WORDS * 4);
   words[APEX_COPY_EXTENT] = width;
   words[APEX_COPY_EXTENT + 1] = height;
   words[APEX_COPY_EXTENT + 2] = planes;
   words[APEX_COPY_BYTES] = bytes;
   words[APEX_COPY_SRC_OFFSET] = src_offset;
   words[APEX_COPY_DST_OFFSET] = dst_offset;
   words[APEX_COPY_MASK] = mask;
   /* Four elements along x per invocation where they lie a texel apart. */
   bool run4 = bytes >= 4 && mask == ~0u &&
               (!(src[APEX_SURFACE_FLAGS] & APEX_SURFACE_TILED) || !(src[APEX_SURFACE_ORIGIN] & 3)) &&
               (!(dst[APEX_SURFACE_FLAGS] & APEX_SURFACE_TILED) || !(dst[APEX_SURFACE_ORIGIN] & 3));
   words[APEX_COPY_RUN] = run4 ? 4 : 1;
   internal_dispatch(cmd, APEX_INTERNAL_COPY, words,
                     (uint32_t[3]){DIV_ROUND_UP(width, run4 ? 128 : 32), height, planes}, 0, false);
}

/* Element of one aspect within a texel: byte offset, bytes and the bits of
 * the destination element it writes when copying into the image. */
static void
aspect_element(VkFormat format, VkImageAspectFlags aspect, unsigned *offset, unsigned *bytes, uint32_t *mask)
{
   *offset = 0;
   *bytes = vk_format_get_blocksize(format);
   *mask = ~0u;
   if (!vk_format_has_depth(format) || !vk_format_has_stencil(format))
      return;
   bool stencil = aspect == VK_IMAGE_ASPECT_STENCIL_BIT;
   if (format == VK_FORMAT_D24_UNORM_S8_UINT) {
      *offset = stencil ? 3 : 0;
      *bytes = stencil ? 1 : 4;
      *mask = stencil ? 0xff : 0x00ffffff;
   } else {
      *offset = stencil ? 4 : 0;
      *bytes = stencil ? 1 : 4;
   }
}

/* Texels of 3, 6 or 12 bytes copy as three elements of a third of a texel. */
static unsigned
element_split(unsigned *bytes)
{
   if (util_is_power_of_two_nonzero(*bytes))
      return 1;
   *bytes /= 3;
   return 3;
}

static void record_decode(struct apex_command_buffer *cmd, const struct apex_image *image, unsigned level,
                          unsigned layer, unsigned layers, VkOffset3D offset, VkExtent3D extent);

/* The format, layout and address of the plane an aspect names: a plane of a
 * multi-planar image, or the image itself. */
static VkFormat
aspect_plane(const struct apex_image *image, VkImageAspectFlags aspect, const struct apex_hw_layout **layout,
             uint64_t *va)
{
   if (vk_format_get_plane_count(image->vk.format) > 1) {
      unsigned plane = aspect & VK_IMAGE_ASPECT_PLANE_1_BIT ? 1 : aspect & VK_IMAGE_ASPECT_PLANE_2_BIT ? 2 : 0;
      *layout = apex_image_plane(image, plane, va);
      return vk_format_get_plane_format(image->vk.format, plane);
   }
   *layout = &image->layout;
   *va = apex_image_va(image);
   return image->vk.format;
}

static void
copy_buffer_image(struct apex_command_buffer *cmd, struct apex_buffer *buffer, struct apex_image *image,
                  unsigned count, const VkBufferImageCopy2 *regions, bool to_image)
{
   for (unsigned i = 0; i < count; i++) {
      const VkBufferImageCopy2 *r = &regions[i];
      struct vk_image_buffer_layout layout = vk_image_buffer_copy_layout(&image->vk, r);
      unsigned l = r->imageSubresource.mipLevel;
      bool three_d = image->vk.image_type == VK_IMAGE_TYPE_3D;
      unsigned first = three_d ? r->imageOffset.z : r->imageSubresource.baseArrayLayer;
      unsigned planes = three_d ? r->imageExtent.depth : vk_image_subresource_layer_count(&image->vk, &r->imageSubresource);
      const struct apex_hw_layout *hw;
      uint64_t va;
      VkFormat format = aspect_plane(image, r->imageSubresource.aspectMask, &hw, &va);
      unsigned bw = vk_format_get_blockwidth(format), bh = vk_format_get_blockheight(format);
      unsigned width = DIV_ROUND_UP(r->imageExtent.width, bw), height = DIV_ROUND_UP(r->imageExtent.height, bh);
      unsigned offset, bytes;
      uint32_t mask;
      aspect_element(format, r->imageSubresource.aspectMask, &offset, &bytes, &mask);
      unsigned split = element_split(&bytes);
      uint32_t image_words[APEX_SURFACE_WORDS], buffer_words[APEX_SURFACE_WORDS];
      image_surface(image, hw, va, l, first, r->imageOffset.x / bw * split, r->imageOffset.y / bh, image_words);
      unsigned element = vk_format_has_depth(format) && vk_format_has_stencil(format) ? bytes :
                         vk_format_get_blocksize(format) / split;
      image_words[APEX_SURFACE_TEXEL] = hw->bytes / split;
      buffer_surface(buffer->vk.device_address + r->bufferOffset, layout.row_stride_B, layout.image_stride_B,
                     element, buffer_words);
      if (to_image)
         copy_surfaces(cmd, buffer_words, image_words, width * split, height, planes, bytes, 0, offset, mask);
      else
         copy_surfaces(cmd, image_words, buffer_words, width * split, height, planes, bytes, offset, 0, ~0u);
      if (to_image)
         record_decode(cmd, image, l, first, planes, r->imageOffset, r->imageExtent);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyBufferToImage2(VkCommandBuffer handle, const VkCopyBufferToImageInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, buffer, info->srcBuffer);
   VK_FROM_HANDLE(apex_image, image, info->dstImage);
   copy_buffer_image(cmd, buffer, image, info->regionCount, info->pRegions, true);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyImageToBuffer2(VkCommandBuffer handle, const VkCopyImageToBufferInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, image, info->srcImage);
   VK_FROM_HANDLE(apex_buffer, buffer, info->dstBuffer);
   copy_buffer_image(cmd, buffer, image, info->regionCount, info->pRegions, false);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyImage2(VkCommandBuffer handle, const VkCopyImageInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, src, info->srcImage);
   VK_FROM_HANDLE(apex_image, dst, info->dstImage);
   for (unsigned i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];
      bool src_3d = src->vk.image_type == VK_IMAGE_TYPE_3D, dst_3d = dst->vk.image_type == VK_IMAGE_TYPE_3D;
      unsigned src_first = src_3d ? r->srcOffset.z : r->srcSubresource.baseArrayLayer;
      unsigned dst_first = dst_3d ? r->dstOffset.z : r->dstSubresource.baseArrayLayer;
      /* 2D arrays and 3D images exchange layers for depth slices. */
      unsigned planes = src_3d ? r->extent.depth : vk_image_subresource_layer_count(&src->vk, &r->srcSubresource);
      const struct apex_hw_layout *src_hw, *dst_hw;
      uint64_t src_va, dst_va;
      VkFormat format = aspect_plane(src, r->srcSubresource.aspectMask, &src_hw, &src_va);
      VkFormat dst_format = aspect_plane(dst, r->dstSubresource.aspectMask, &dst_hw, &dst_va);
      unsigned bw = vk_format_get_blockwidth(format), bh = vk_format_get_blockheight(format);
      unsigned dbw = vk_format_get_blockwidth(dst_format), dbh = vk_format_get_blockheight(dst_format);
      unsigned width = DIV_ROUND_UP(r->extent.width, bw), height = DIV_ROUND_UP(r->extent.height, bh);
      unsigned offset, bytes, dst_offset, unused;
      uint32_t mask;
      aspect_element(format, r->srcSubresource.aspectMask, &offset, &bytes, &mask);
      aspect_element(dst_format, r->dstSubresource.aspectMask, &dst_offset, &unused, &mask);
      unsigned split = element_split(&bytes);
      uint32_t from[APEX_SURFACE_WORDS], to[APEX_SURFACE_WORDS];
      image_surface(src, src_hw, src_va, r->srcSubresource.mipLevel, src_first,
                    r->srcOffset.x / bw * split, r->srcOffset.y / bh, from);
      image_surface(dst, dst_hw, dst_va, r->dstSubresource.mipLevel, dst_first,
                    r->dstOffset.x / dbw * split, r->dstOffset.y / dbh, to);
      from[APEX_SURFACE_TEXEL] /= split;
      to[APEX_SURFACE_TEXEL] /= split;
      /* Multisampled images copy every sample plane. */
      uint32_t samples = src->vk.samples;
      copy_surfaces(cmd, from, to, width * split, height, planes * samples, bytes, offset, dst_offset, mask);
      record_decode(cmd, dst, r->dstSubresource.mipLevel, dst_first, planes, r->dstOffset,
                         (VkExtent3D){width * dbw, height * dbh, 1});
   }
}

/* Re-decodes an ETC2/EAC or BC region (texels, clamped to the level) into
 * the image's decoded plane after its blocks change. */
static void
record_decode(struct apex_command_buffer *cmd, const struct apex_image *image, unsigned level,
                   unsigned layer, unsigned layers, VkOffset3D offset, VkExtent3D extent)
{
   uint32_t kind;
   if (!apex_decoded_format(image->vk.format, &kind))
      return;
   uint32_t width = MIN2(extent.width, u_minify(image->vk.extent.width, level) - offset.x);
   uint32_t height = MIN2(extent.height, u_minify(image->vk.extent.height, level) - offset.y);
   const struct apex_hw_level *blocks = &image->layout.level[level];
   /* 3D slices step by the level's plane, as in image_surface(). */
   uint64_t slice = image->vk.image_type == VK_IMAGE_TYPE_3D ? blocks->plane : image->layout.layer_stride;
   uint64_t src = apex_image_va(image) + blocks->offset + (uint64_t)layer * slice +
                  (uint64_t)(offset.y / 4) * blocks->pitch + (uint64_t)(offset.x / 4) * image->layout.bytes;
   uint32_t words[APEX_JOB_WORDS] = {0};
   image_surface(image, &image->decoded_layout, apex_image_va(image) + image->decoded, level, layer,
                 offset.x, offset.y, &words[APEX_DECODE_DST]);
   words[APEX_DECODE_SRC] = LO(src);
   words[APEX_DECODE_SRC + 1] = HI(src);
   words[APEX_DECODE_SRC_ROW] = blocks->pitch;
   words[APEX_DECODE_SRC_SLICE] = slice;
   words[APEX_DECODE_EXTENT] = width;
   words[APEX_DECODE_EXTENT + 1] = height;
   words[APEX_DECODE_EXTENT + 2] = layers;
   words[APEX_DECODE_KIND] = kind;
   /* Dispatches overlap, so the decode waits for the copy of its blocks. */
   apex_cp_barrier(&cmd->ib, APEX_CP_CLASS_COMPUTE, APEX_CP_CACHE_L1);
   /* The BC kernel takes a block per invocation, the ETC2 kernel a texel. */
   if (kind >= APEX_BC1_RGB)
      internal_dispatch(cmd, APEX_INTERNAL_BC, words,
                        (uint32_t[3]){DIV_ROUND_UP(width, 128), DIV_ROUND_UP(height, 4), layers}, 0, false);
   else
      internal_dispatch(cmd, APEX_INTERNAL_ETC2, words,
                        (uint32_t[3]){DIV_ROUND_UP(width, 32), height, layers}, 0, false);
}

/* The clear kernel over layers x rect of one level: pattern and mask
 * words of the packed texel. */
static void
record_clear(struct apex_command_buffer *cmd, const struct apex_image *image, unsigned level,
             unsigned layer, unsigned layers, const uint32_t pattern[4], const uint32_t mask[4])
{
   unsigned width = u_minify(image->vk.extent.width, level);
   unsigned height = u_minify(image->vk.extent.height, level);
   if (image->vk.image_type == VK_IMAGE_TYPE_3D) {
      /* Every depth slice of a 3D level is a plane. */
      layer = 0;
      layers = u_minify(image->vk.extent.depth, level);
   }
   uint32_t words[APEX_JOB_WORDS] = {0};
   image_surface(image, &image->layout, apex_image_va(image), level, layer, 0, 0, &words[APEX_CLEAR_DST]);
   unsigned planes = layers * (image->vk.image_type == VK_IMAGE_TYPE_3D ? 1 : image->vk.samples);
   words[APEX_CLEAR_EXTENT] = width;
   words[APEX_CLEAR_EXTENT + 1] = height;
   words[APEX_CLEAR_EXTENT + 2] = planes;
   words[APEX_CLEAR_BYTES] = vk_format_get_blocksize(image->vk.format);
   memcpy(&words[APEX_CLEAR_PATTERN], pattern, 16);
   memcpy(&words[APEX_CLEAR_MASK], mask, 16);
   internal_dispatch(cmd, APEX_INTERNAL_CLEAR, words, (uint32_t[3]){DIV_ROUND_UP(width, 32), height, planes}, 0, false);
}

static void
clear_ranges(struct apex_command_buffer *cmd, struct apex_image *image, uint32_t count,
             const VkImageSubresourceRange *ranges, const uint32_t pattern[4], const uint32_t mask[4])
{
   for (unsigned i = 0; i < count; i++) {
      unsigned levels = vk_image_subresource_level_count(&image->vk, &ranges[i]);
      unsigned layers = vk_image_subresource_layer_count(&image->vk, &ranges[i]);
      for (unsigned l = ranges[i].baseMipLevel; l < ranges[i].baseMipLevel + levels; l++)
         record_clear(cmd, image, l, ranges[i].baseArrayLayer, layers, pattern, mask);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdClearColorImage(VkCommandBuffer handle, VkImage img, VkImageLayout layout,
                        const VkClearColorValue *color, uint32_t count, const VkImageSubresourceRange *ranges)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, image, img);
   uint32_t pattern[4], mask[4] = {~0u, ~0u, ~0u, ~0u};
   pack_color(image->vk.format, color, pattern);
   clear_ranges(cmd, image, count, ranges, pattern, mask);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdClearDepthStencilImage(VkCommandBuffer handle, VkImage img, VkImageLayout layout,
                               const VkClearDepthStencilValue *value, uint32_t count,
                               const VkImageSubresourceRange *ranges)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, image, img);
   VkFormat format = image->vk.format;
   for (unsigned i = 0; i < count; i++) {
      uint32_t pattern[4] = {0}, mask[4] = {0};
      pack_depth_stencil(format, value->depth, value->stencil, pattern);
      /* Each aspect writes its bits of the texel. */
      for (unsigned a = 0; a < 2; a++) {
         VkImageAspectFlags aspect = a ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
         if (!(ranges[i].aspectMask & aspect) || !(vk_format_aspects(format) & aspect))
            continue;
         unsigned offset, bytes;
         uint32_t bits;
         aspect_element(format, aspect, &offset, &bytes, &bits);
         if (format == VK_FORMAT_D24_UNORM_S8_UINT)
            mask[0] |= a ? 0xff000000u : 0x00ffffffu;
         else
            mask[offset / 4] |= bytes == 4 ? ~0u : BITFIELD_MASK(bytes * 8);
      }
      clear_ranges(cmd, image, 1, &ranges[i], pattern, mask);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBlitImage2(VkCommandBuffer handle, const VkBlitImageInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct saved_state saved;
   meta_begin(cmd, &saved);
   vk_meta_blit_image2(&cmd->vk, &device->meta, info);
   meta_end(cmd, &saved);
}

/* Resolves run as passes that load the multisampled region and resolve
 * it from the tile buffer when source and destination align; vk_meta
 * resolves the others by sample fetches. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdResolveImage2(VkCommandBuffer handle, const VkResolveImageInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, src, info->srcImage);
   VK_FROM_HANDLE(apex_image, dst, info->dstImage);
   struct apex_device *device = (void *)cmd->vk.base.device;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageResolve2 *r = &info->pRegions[i];
      uint8_t format = apex_hw_color_format(src->vk.format);
      unsigned layers = vk_image_subresource_layer_count(&src->vk, &r->srcSubresource);
      if (src->vk.format != dst->vk.format || !format || r->srcOffset.x != r->dstOffset.x ||
          r->srcOffset.y != r->dstOffset.y || src->vk.image_type != VK_IMAGE_TYPE_2D ||
          dst->vk.image_type != VK_IMAGE_TYPE_2D || cmd->rendering.active) {
         VkResolveImageInfo2 one = *info;
         one.regionCount = 1;
         one.pRegions = r;
         struct saved_state saved;
         meta_begin(cmd, &saved);
         vk_meta_resolve_image2(&cmd->vk, &device->meta, &one);
         meta_end(cmd, &saved);
         continue;
      }
      const struct apex_attachment from = {
         .image = src, .format = src->vk.format, .level = r->srcSubresource.mipLevel,
         .layer = r->srcSubresource.baseArrayLayer, .aspects = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      const struct apex_attachment to = {
         .image = dst, .format = dst->vk.format, .level = r->dstSubresource.mipLevel,
         .layer = r->dstSubresource.baseArrayLayer, .aspects = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      memset(&cmd->rendering, 0, sizeof(cmd->rendering));
      struct apex_hw_pass *pass = &cmd->rendering.pass;
      pass->x0 = MAX2(r->dstOffset.x, 0);
      pass->y0 = MAX2(r->dstOffset.y, 0);
      pass->x1 = MIN2(pass->x0 + r->extent.width, 0xffff);
      pass->y1 = MIN2(pass->y0 + r->extent.height, 0xffff);
      pass->layers = layers;
      pass->samples = src->vk.samples;
      struct apex_hw_attachment *a = &pass->attachment[0];
      a->present = a->resolve = true;
      a->load = APEX_HW_LOAD;
      a->format = format;
      a->planes = format_planes(src->vk.format);
      a->surface = surface_of(&from);
      a->resolve_surface = surface_of(&to);
      emit_pass(cmd);
      end_pass(cmd);
   }
}

/* ---- Barriers ----------------------------------------------------------- */

/* Transfers run on the copy engine, as compute kernels and as passes. */
static uint32_t
barrier_classes(VkPipelineStageFlags2 stages)
{
   if (stages & (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT))
      return APEX_CP_CLASS_ALL;
   uint32_t classes = 0;
   /* Transform feedback captures in compute programs. */
   if (stages & (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT))
      classes |= APEX_CP_CLASS_COMPUTE;
   if (stages & (VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                 VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                 VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT))
      classes |= APEX_CP_CLASS_GRAPHICS;
   if (stages & (VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT |
                 VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
                 VK_PIPELINE_STAGE_2_BLIT_BIT))
      classes |= APEX_CP_CLASS_ALL;
   return classes;
}

/* Waits for the source classes and invalidates the L1 and texture caches
 * for device-side destinations. Inside a pass, draws and the attachments
 * they write are ordered by the hardware. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdPipelineBarrier2(VkCommandBuffer handle, const VkDependencyInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (cmd->rendering.active) {
      split_pass(cmd);
      return;
   }
   VkPipelineStageFlags2 src = 0;
   VkAccessFlags2 access = 0;
   for (uint32_t i = 0; i < info->memoryBarrierCount; i++) {
      src |= info->pMemoryBarriers[i].srcStageMask;
      access |= info->pMemoryBarriers[i].dstAccessMask;
   }
   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++) {
      src |= info->pBufferMemoryBarriers[i].srcStageMask;
      access |= info->pBufferMemoryBarriers[i].dstAccessMask;
   }
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
      src |= info->pImageMemoryBarriers[i].srcStageMask;
      access |= info->pImageMemoryBarriers[i].dstAccessMask;
   }
   uint32_t classes = barrier_classes(src) & cmd->pending;
   uint32_t cache = access & ~(VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT) ?
      APEX_CP_CACHE_L1 | APEX_CP_CACHE_TEXTURE : 0;
   if (!classes && !cache)
      return;
   apex_cp_barrier(&cmd->ib, classes, cache);
   cmd->pending &= ~classes;
   if (classes & APEX_CP_CLASS_COMPUTE)
      cmd->private_pending = false;
}

/* ---- vk_meta uploads ---------------------------------------------------- */

/* Recording snapshots host data and reserves an address without kernel waits.
 * Submission materializes this immutable source before publishing dependencies. */
static VkResult
bind_map_upload(struct vk_command_buffer *vk, struct vk_meta_device *meta,
                VkBuffer handle, void **map_out)
{
   struct apex_command_buffer *cmd = (void *)vk;
   struct apex_device *device = (void *)vk->base.device;
   VK_FROM_HANDLE(vk_buffer, buffer, handle);
   struct apex_upload *upload = vk_zalloc(&vk->pool->alloc, sizeof(*upload) + buffer->size,
                                          8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!upload)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   upload->size = buffer->size;
   mtx_lock(&device->va_mutex);
   upload->reserved_va = util_vma_heap_alloc(&device->va_heap, align64(upload->size, 4096), 4096);
   mtx_unlock(&device->va_mutex);
   if (!upload->reserved_va) {
      vk_free(&vk->pool->alloc, upload);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   buffer->device_address = upload->reserved_va;
   *map_out = upload->data;
   list_addtail(&upload->link, &cmd->uploads);
   return VK_SUCCESS;
}

/* ---- Submission --------------------------------------------------------- */

/* Table layout: (descriptor_count + 1) rows, the push image, the trailer. */
static uint64_t
table_bytes(const struct apex_table *table)
{
   return apex_program_trailer(table->programs[0]) + sizeof(table->trailer);
}

static bool
row_used(const struct apex_table *table, unsigned row)
{
   for (unsigned p = 0; p < 2; p++)
      if (table->programs[p] && BITSET_TEST(table->programs[p]->used_descriptors, row))
         return true;
   return false;
}

/* Writes a table into submission memory at `map`, GPUVA `va`. */
static VkResult
write_table(struct apex_device *device, const struct apex_table *table, void *map, uint64_t va,
            uint64_t data_va, uint64_t scratch_va)
{
   struct apex_program *program = table->programs[0];
   if (!program->table)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   size_t push_offset = (program->descriptor_count + 1) * sizeof(union apex_descriptor);
   size_t trailer_offset = apex_program_trailer(program);
   size_t bytes = table_bytes(table);
   union apex_descriptor *rows = calloc(1, bytes);
   if (!rows)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   /* Null and sentinel rows are empty robust buffers; image rows overwrite. */
   for (unsigned r = 0; r <= program->descriptor_count; r++)
      rows[r].buffer.flags = util_cpu_to_le32(APEX_BUFFER_ROBUST);
   memcpy((uint8_t *)rows + push_offset, table->push, program->push_size);
   uint32_t *trailer = (void *)((uint8_t *)rows + trailer_offset);
   for (unsigned w = 0; w < ARRAY_SIZE(table->trailer); w++)
      trailer[w] = util_cpu_to_le32(table->trailer[w]);
   u_foreach_bit64(w, table->relocs | table->scratch_relocs) {
      uint64_t base = table->scratch_relocs & BITFIELD64_BIT(w) ? scratch_va : data_va;
      uint64_t target = base + (table->trailer[w] | (uint64_t)table->trailer[w + 1] << 32);
      trailer[w] = util_cpu_to_le32(LO(target));
      trailer[w + 1] = util_cpu_to_le32(HI(target));
   }
   /* Set rows copy into the table; an update-after-bind binding's row
    * addresses its rows in pool memory. */
   VkResult result = VK_ERROR_DEVICE_LOST;
   for (unsigned s = 0; s < program->set_count; s++) {
      const struct apex_set_layout *layout = program->set_layouts[s];
      if (!layout || !layout->slot_count)
         continue;
      const struct apex_bound_set *bound = table->sets[s];
      const struct apex_descriptor_set *set = bound ? bound->set : NULL;
      for (unsigned b = 0; b < layout->binding_count; b++) {
         const struct apex_binding_layout *binding = &layout->bindings[b];
         unsigned slots = binding->slots;
         unsigned base = program->set_offsets[s] + binding->slot;
         unsigned elements = !binding->count ? 0 :
                             apex_binding_in_memory(binding) || binding->bytes ? 1 : binding->count;
         for (unsigned e = 0; e < elements; e++) {
            unsigned row = base + e * slots;
            if (!row_used(table, row))
               continue;
            if (!set || memcmp(set->layout->vk.blake3, layout->vk.blake3, BLAKE3_OUT_LEN))
               goto out;
            uint32_t count = apex_set_binding_count(set, binding);
            if (apex_binding_in_memory(binding)) {
               uint64_t va = set->va + binding->row * sizeof(union apex_descriptor);
               rows[row].buffer = (struct apex_buffer_descriptor){
                  util_cpu_to_le32(LO(va)), util_cpu_to_le32(HI(va)),
                  util_cpu_to_le32(count * slots * sizeof(union apex_descriptor)),
                  util_cpu_to_le32(APEX_BUFFER_ROBUST),
               };
               continue;
            }
            if (binding->bytes) {
               /* The buffer row is completed with the table address below. */
               memcpy(&rows[row], &set->rows[binding->row],
                      (count ? 1 + DIV_ROUND_UP(count, sizeof(union apex_descriptor)) : 0) * sizeof(rows[0]));
               continue;
            }
            if (e >= count)
               continue;
            memcpy(&rows[row], &set->rows[binding->row + e * slots], slots * sizeof(rows[0]));
            if (vk_descriptor_type_is_dynamic(binding->type)) {
               uint32_t offset = bound->offsets[binding->dynamic + e];
               uint64_t va = (util_le32_to_cpu(rows[row].buffer.low) |
                              (uint64_t)util_le32_to_cpu(rows[row].buffer.high) << 32) + offset;
               rows[row].buffer.low = util_cpu_to_le32(LO(va));
               rows[row].buffer.high = util_cpu_to_le32(HI(va));
               if (rows[row].words[4])
                  rows[row].buffer.bytes = util_cpu_to_le32(util_le32_to_cpu(rows[row].buffer.bytes) - offset);
               rows[row].words[4] = 0;
            }
         }
      }
   }
   /* Inline uniform block rows address their data rows in the table. */
   for (unsigned s = 0; s < program->set_count; s++) {
      const struct apex_set_layout *layout = program->set_layouts[s];
      for (unsigned b = 0; layout && b < layout->binding_count; b++) {
         unsigned row = program->set_offsets[s] + layout->bindings[b].slot;
         if (!layout->bindings[b].bytes || !row_used(table, row))
            continue;
         uint64_t inline_va = va + (row + 1) * sizeof(union apex_descriptor);
         rows[row].buffer.low = util_cpu_to_le32(LO(inline_va));
         rows[row].buffer.high = util_cpu_to_le32(HI(inline_va));
      }
   }
   memcpy(map, rows, bytes);
   result = VK_SUCCESS;
out:
   free(rows);
   return result;
}

VkResult
apex_cmd_prepare(struct apex_device *device, struct apex_queue *queue,
                 struct apex_command_buffer *cmd, struct apex_arena **out, uint64_t *ib_va)
{
   /* Arena: the data blocks, the tables, the IB, then the scratch. */
   uint64_t bytes = align64(cmd->data.size, 64);
   list_for_each_entry(struct apex_table, table, &cmd->tables, link) {
      table->offset = bytes;
      bytes += align64(table_bytes(table), 64);
   }
   uint64_t ib_offset = bytes;
   bytes += (uint64_t)cmd->ib.count * 4;
   uint64_t scratch_offset = align64(bytes, 256);
   if (cmd->scratch)
      bytes = scratch_offset + cmd->scratch;
   struct apex_arena *arena = apex_arena_get(device, queue, bytes);
   if (!arena)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   list_addtail(&arena->link, &queue->busy_arenas);
   arena->sequence = UINT64_MAX;
   uint8_t *map = arena->bo.map;
   uint64_t va = arena->bo.va;
   if (cmd->data.size)
      memcpy(map, cmd->data.data, cmd->data.size);
   list_for_each_entry(struct apex_table, table, &cmd->tables, link) {
      VkResult result = write_table(device, table, map + table->offset, va + table->offset, va,
                                    va + scratch_offset);
      if (result != VK_SUCCESS)
         return result;
   }
   uint32_t *ib = (uint32_t *)(map + ib_offset);
   memcpy(ib, cmd->ib.words, (size_t)cmd->ib.count * 4);
   util_dynarray_foreach(&cmd->patches, struct apex_patch, patch) {
      uint64_t target;
      switch (patch->kind) {
      case APEX_PATCH_TABLE:
         target = va + patch->table->offset;
         break;
      case APEX_PATCH_GRID:
         target = va + patch->table->offset + apex_program_trailer(patch->table->programs[0]);
         break;
      case APEX_PATCH_PROGRAM: {
         VkResult result = apex_program_upload(device, patch->program);
         if (result != VK_SUCCESS)
            return result;
         target = patch->program->bo.va;
         break;
      }
      case APEX_PATCH_SCRATCH:
         target = va + scratch_offset + patch->offset;
         break;
      case APEX_PATCH_IB:
         target = va + ib_offset + patch->offset * 4;
         break;
      default:
         target = va + patch->offset;
         break;
      }
      ib[patch->dword] = LO(target);
      ib[patch->dword + 1] = HI(target);
   }
   *out = arena;
   *ib_va = va + ib_offset;
   return VK_SUCCESS;
}

void
apex_cmd_entrypoints(struct vk_device_entrypoint_table *t)
{
   t->BeginCommandBuffer = apex_BeginCommandBuffer;
   t->EndCommandBuffer = apex_EndCommandBuffer;
   t->CmdBindPipeline = apex_CmdBindPipeline;
   t->CmdBindDescriptorSets2 = apex_CmdBindDescriptorSets2;
   t->CmdPushDescriptorSet2 = apex_CmdPushDescriptorSet2;
   t->CmdPushDescriptorSetWithTemplate2 = apex_CmdPushDescriptorSetWithTemplate2;
   t->CmdPushConstants2 = apex_CmdPushConstants2;
   t->CmdDispatch = apex_CmdDispatch;
   t->CmdDispatchBase = apex_CmdDispatchBase;
   t->CmdDispatchIndirect = apex_CmdDispatchIndirect;
   t->CmdBeginRendering = apex_CmdBeginRendering;
   t->CmdEndRendering = apex_CmdEndRendering;
   t->CmdClearAttachments = apex_CmdClearAttachments;
   t->CmdBindVertexBuffers2 = apex_CmdBindVertexBuffers2;
   t->CmdBindIndexBuffer2 = apex_CmdBindIndexBuffer2;
   t->CmdDraw = apex_CmdDraw;
   t->CmdDrawIndexed = apex_CmdDrawIndexed;
   t->CmdDrawIndirect = apex_CmdDrawIndirect;
   t->CmdDrawIndexedIndirect = apex_CmdDrawIndexedIndirect;
   t->CmdDrawIndirectCount = apex_CmdDrawIndirectCount;
   t->CmdDrawIndexedIndirectCount = apex_CmdDrawIndexedIndirectCount;
   t->CmdSetEvent2 = apex_CmdSetEvent2;
   t->CmdResetEvent2 = apex_CmdResetEvent2;
   t->CmdWaitEvents2 = apex_CmdWaitEvents2;
   t->CmdResetQueryPool = apex_CmdResetQueryPool;
   t->CmdBindTransformFeedbackBuffersEXT = apex_CmdBindTransformFeedbackBuffersEXT;
   t->CmdBeginTransformFeedbackEXT = apex_CmdBeginTransformFeedbackEXT;
   t->CmdEndTransformFeedbackEXT = apex_CmdEndTransformFeedbackEXT;
   t->CmdBeginQueryIndexedEXT = apex_CmdBeginQueryIndexedEXT;
   t->CmdEndQueryIndexedEXT = apex_CmdEndQueryIndexedEXT;
   t->CmdWriteTimestamp2 = apex_CmdWriteTimestamp2;
   t->CmdCopyQueryPoolResults = apex_CmdCopyQueryPoolResults;
   t->CmdFillBuffer = apex_CmdFillBuffer;
   t->CmdCopyBuffer2 = apex_CmdCopyBuffer2;
   t->CmdUpdateBuffer = apex_CmdUpdateBuffer;
   t->CmdCopyBufferToImage2 = apex_CmdCopyBufferToImage2;
   t->CmdCopyImageToBuffer2 = apex_CmdCopyImageToBuffer2;
   t->CmdCopyImage2 = apex_CmdCopyImage2;
   t->CmdClearColorImage = apex_CmdClearColorImage;
   t->CmdClearDepthStencilImage = apex_CmdClearDepthStencilImage;
   t->CmdBlitImage2 = apex_CmdBlitImage2;
   t->CmdResolveImage2 = apex_CmdResolveImage2;
   t->CmdPipelineBarrier2 = apex_CmdPipelineBarrier2;
   t->CmdBeginConditionalRenderingEXT = apex_CmdBeginConditionalRenderingEXT;
   t->CmdEndConditionalRenderingEXT = apex_CmdEndConditionalRenderingEXT;
}

void
apex_cmd_meta_init(struct vk_meta_device *meta)
{
   meta->cmd_bind_map_buffer = bind_map_upload;
   /* vk_meta splits rectangle draws into vertex uploads of this size. */
   meta->max_bind_map_buffer_size_B = 64 * 1024;
}
