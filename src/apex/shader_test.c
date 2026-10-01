/* SPDX-License-Identifier: MIT */
/* Compiles the benchmark, triangle and compositor shaders and the captured
 * GNOME Shell and glmark2 shaders, runs them on the ISA reference model
 * (Tooling/apex_isa) against host references, and holds the zero-spill gate.
 * Usage: apex-shader-test DIR GNOME_DIR GLMARK2_DIR where DIR holds the
 * SPIR-V of bench.{comp,vert,frag}, triangle.{vert,frag} and compositor.frag.
 * The model's texture unit returns coordinate register a + k plus image and
 * sampler descriptor dwords k and the request flags as component k, which
 * checks coordinate placement and the instruction's texture fields. */
#include "apex.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

#define ROOT 0x100000ull
#define ROOT_BYTES (64u * 2049u + 256u)
#define PUSH (64u * 2049u)
static unsigned slot(unsigned set, unsigned binding) { return 64 * (1 + 256 * set + binding); }

struct world {
   struct apex_sim_region r[8];
   unsigned n;
};
static uint8_t *region(struct world *w, uint64_t va, size_t size)
{
   CHECK(w->n < 8);
   uint8_t *data = calloc(1, size);
   w->r[w->n++] = (struct apex_sim_region){va, data, size};
   return data;
}
static void world_free(struct world *w)
{
   for (unsigned i = 0; i < w->n; i++)
      free(w->r[i].data);
}
static void put(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static uint32_t get(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint32_t bits(float f) { uint32_t v; memcpy(&v, &f, 4); return v; }
static float flt(uint32_t v) { float f; memcpy(&f, &v, 4); return f; }

static void buffer_descriptor(uint8_t *p, uint64_t va, uint32_t bytes)
{
   put(p, (uint32_t)va);
   put(p + 4, va >> 32);
   put(p + 8, bytes);
   put(p + 12, 1); /* out-of-range loads read zero, stores drop */
}
static void set_bits(uint8_t *p, unsigned lo, unsigned width, uint64_t v)
{
   for (unsigned k = 0; k < width; k++)
      if (v >> k & 1)
         p[(lo + k) / 8] |= 1u << ((lo + k) % 8);
}
/* Linear RGBA8 UNORM 2D image with one level (architecture, Texture unit
 * descriptor bits), and a nearest, repeating sampler (all fields zero). */
static void image_descriptor(uint8_t *p, uint64_t va, unsigned w, unsigned h)
{
   memset(p, 0, 32);
   set_bits(p, 0, 40, va);
   set_bits(p, 40, 4, 3);
   set_bits(p, 49, 3, 1);
   set_bits(p, 55, 12, 0 | 1 << 3 | 2 << 6 | 3 << 9);
   set_bits(p, 67, 14, w - 1);
   set_bits(p, 81, 14, h - 1);
   set_bits(p, 106, 5, 1);
   set_bits(p, 111, 16, (w * 4 + 63) / 64);
}
static void sampler_descriptor(uint8_t *p)
{
   memset(p, 0, 32);
}
/* Component k of the model's texture result for coordinate bits x
 * (apex_isa default_texture): the request flags hold the texel offsets in
 * 4-bit fields, the variant at 13 (fetch 4), the dimensionality at 16 and the
 * gather component at 19. */
static uint32_t texel(const uint8_t *image, const uint8_t *sampler, unsigned k, uint32_t x, uint32_t flags)
{
   return x + get(image + 4 * k) + get(sampler + 4 * k) + flags;
}
#define FLAGS_2D (1u << 16)

static uint32_t *read_file(const char *path, size_t *words)
{
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long size = ftell(f);
   CHECK(size > 0 && size % 4 == 0);
   rewind(f);
   uint32_t *data = malloc(size);
   CHECK(data && fread(data, 1, size, f) == (size_t)size && !fclose(f));
   *words = size / 4;
   return data;
}

static unsigned total_instructions, max_vector, max_scalar, shaders;

static void compile_spilling(const char *dir, const char *name, struct apex_compile_result *r,
                             bool spills)
{
   char path[4096];
   snprintf(path, sizeof(path), "%s/%s", dir, name);
   size_t words;
   uint32_t *spirv = read_file(path, &words);
   if (apex_compile_spirv(spirv, words, r)) {
      fprintf(stderr, "%s: %s\n", name, r->diagnostic);
      abort();
   }
   free(spirv);
   printf("%-24s %4u instructions  s%-3u v%-3u spills %u\n", name, r->instructions, r->scalar,
          r->vector, r->spills);
   CHECK(!r->spills == !spills && r->vector <= 128 && r->scalar <= 96);
   total_instructions += r->instructions;
   max_vector = r->vector > max_vector ? r->vector : max_vector;
   max_scalar = r->scalar > max_scalar ? r->scalar : max_scalar;
   shaders++;
}

static void compile(const char *dir, const char *name, struct apex_compile_result *r)
{
   compile_spilling(dir, name, r, false);
}

static void run_private(const struct apex_compile_result *r, struct world *w, const uint32_t user[16],
                        unsigned groups, struct apex_sim_wave *wave, uint64_t private_base)
{
   uint32_t grid[3] = {groups, 1, 1};
   char diagnostic[256] = "";
   if (apex_simulate(r->data, r->size, user, grid, private_base, wave, w->r, w->n, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
}
static void run(const struct apex_compile_result *r, struct world *w, const uint32_t user[16],
                unsigned groups, struct apex_sim_wave *wave)
{
   run_private(r, w, user, groups, wave, 0);
}

static void root_user(uint32_t user[16])
{
   memset(user, 0, 16 * sizeof(*user));
   user[0] = (uint32_t)ROOT;
   user[1] = ROOT >> 32;
}

/* bench.comp: streaming read (xor reduction), write and copy. */
static void test_bench_compute(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "bench.comp.spv", &r);
   const unsigned groups = 3, stride = groups * 256, count = 2000;
   for (unsigned mode = 0; mode < 3; mode++) {
      struct world w = {0};
      uint8_t *root = region(&w, ROOT, ROOT_BYTES);
      uint8_t *src = region(&w, 0x200000, count * 16);
      uint8_t *dst = region(&w, 0x400000, count * 16);
      for (unsigned i = 0; i < count * 4; i++)
         put(src + 4 * i, 0x9e3779b9u * (i + 1));
      buffer_descriptor(root + slot(0, 0), 0x200000, count * 16);
      buffer_descriptor(root + slot(0, 1), 0x400000, count * 16);
      put(root + PUSH, mode);
      put(root + PUSH + 4, count);
      uint32_t user[16];
      root_user(user);
      user[2] = groups, user[3] = user[4] = 1;
      run(&r, &w, user, groups, NULL);
      for (unsigned i = 0; i < count; i++) {
         for (unsigned c = 0; c < 4; c++) {
            uint32_t want;
            if (mode == 0) {
               want = 0;
               if (i < stride)
                  for (unsigned k = i; k < count; k += stride)
                     want ^= get(src + 16 * k + 4 * c);
               if (i >= stride)
                  want = 0;
            } else {
               want = mode == 1 ? i : get(src + 16 * i + 4 * c);
            }
            CHECK(get(dst + 16 * i + 4 * c) == want);
         }
      }
      world_free(&w);
   }
   apex_compile_result_finish(&r);
}

static void vertex_inputs(struct apex_sim_wave *wave, unsigned location, unsigned n, float base)
{
   for (unsigned c = 0; c < n; c++)
      for (unsigned l = 0; l < 16; l++)
         wave->vector[2 + 4 * location + c][l] = bits(base + l * 0.25f + c);
}

/* Vertex programs store {position}, {point size, layer, viewport, cull},
 * clip distances, then varying L at record vec4 4 + L, 64 bytes a line. */
static void test_vertex(const char *dir, const char *name, unsigned varying_components)
{
   struct apex_compile_result r;
   compile(dir, name, &r);
   uint32_t stride = r.data[28] | r.data[29] << 8;
   CHECK(stride == 128);
   struct world w = {0};
   uint8_t *out = region(&w, 0x800000, 16 * stride);
   region(&w, ROOT, ROOT_BYTES);
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   wave->exec = 0xffff;
   wave->scalar[0] = (uint32_t)ROOT;
   wave->scalar[16] = 0x800000;
   vertex_inputs(wave, 0, 4, 1.0f);
   vertex_inputs(wave, 1, varying_components, -3.0f);
   run(&r, &w, wave->scalar, 1, wave);
   for (unsigned l = 0; l < 16; l++) {
      for (unsigned c = 0; c < 4; c++)
         CHECK(get(out + l * stride + 4 * c) == wave->vector[2 + c][l]);
      for (unsigned c = 0; c < varying_components; c++)
         CHECK(get(out + l * stride + 64 + 4 * c) == wave->vector[6 + c][l]);
   }
   free(wave);
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* transform.vert: the varying's translation joins its first product
 * (within rounding of the exact sum); the position keeps the association
 * its expression gives, bit for bit. */
static void test_transform(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "transform.vert.spv", &r);
   uint32_t stride = r.data[28] | r.data[29] << 8;
   struct world w = {0};
   uint8_t *out = region(&w, 0x800000, 16 * stride);
   uint8_t *root = region(&w, ROOT, ROOT_BYTES);
   float m[2][4][4];
   for (unsigned k = 0; k < 2; k++)
      for (unsigned c = 0; c < 4; c++)
         for (unsigned e = 0; e < 4; e++) {
            m[k][c][e] = c == 3 ? 1000.3f + 7.1f * e + k : 0.37f * (c + 1) - 0.11f * e + 0.013f * k;
            put(root + PUSH + 64 * k + 16 * c + 4 * e, bits(m[k][c][e]));
         }
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   wave->exec = 0xffff;
   wave->scalar[0] = (uint32_t)ROOT;
   wave->scalar[16] = 0x800000;
   vertex_inputs(wave, 0, 4, 1.0f);
   run(&r, &w, wave->scalar, 1, wave);
   bool differs = false;
   for (unsigned l = 0; l < 16; l++) {
      float x = flt(wave->vector[2][l]), y = flt(wave->vector[3][l]), z = flt(wave->vector[4][l]);
      for (unsigned e = 0; e < 4; e++) {
         const float (*a)[4] = m[0], (*b)[4] = m[1];
         float position = fmaf(a[2][e], z, fmaf(a[0][e], x, a[1][e] * y)) + a[3][e];
         differs |= position != fmaf(a[2][e], z, fmaf(a[0][e], x, fmaf(a[1][e], y, a[3][e])));
         CHECK(get(out + l * stride + 4 * e) == bits(position));
         double exact = (double)b[0][e] * x + (double)b[1][e] * y + (double)b[2][e] * z + b[3][e];
         CHECK(fabs(flt(get(out + l * stride + 64 + 4 * e)) - exact) <= 2e-7 * fabs(exact));
      }
   }
   CHECK(differs);
   free(wave);
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* Fragment waves: i and j per lane in v0-v1, covered pixels of quad q at
 * (2q, 0), attribute block of primitive 0. */
static void fragment_wave(struct apex_sim_wave *wave, uint32_t *attributes)
{
   wave->exec = 0xffff;
   wave->scalar[0] = (uint32_t)ROOT;
   for (unsigned l = 0; l < 16; l++) {
      wave->vector[0][l] = bits(0.03125f * (l & 3) + 0.25f * (l >> 2));
      wave->vector[1][l] = bits(0.0625f * (l >> 2));
      wave->vector[2][l] = 2 * (l >> 2);
      wave->vector[3][l] = 0xff;
   }
   wave->attributes = attributes;
   wave->primitives = 1;
}
/* P0, P10 (k = 1) or P20 (k = 2) of attribute code c in a block. */
static uint32_t *attribute(uint32_t *block, unsigned c, unsigned k)
{
   return block + 8 + 12 * (c / 4) + c % 4 + 4 * k;
}
static float interpolate(uint32_t *attributes, unsigned component, uint32_t i, uint32_t j)
{
   float p0 = flt(*attribute(attributes, component, 0)), p10 = flt(*attribute(attributes, component, 1));
   float p20 = flt(*attribute(attributes, component, 2));
   return fmaf(flt(j), p20, fmaf(flt(i), p10, p0));
}

static void test_fragment(const char *dir, const char *name, bool textured, bool modulated)
{
   struct apex_compile_result r;
   compile(dir, name, &r);
   struct world w = {0};
   uint8_t *root = region(&w, ROOT, ROOT_BYTES);
   const unsigned tw = 8, th = 4;
   uint8_t *image = region(&w, 0x900000, 64 * th);
   for (unsigned k = 0; k < tw * th * 4; k++)
      image[(k / (tw * 4)) * 64 + k % (tw * 4)] = (uint8_t)(k * 37 + 11);
   uint8_t packed[8 * 4 * 4];
   for (unsigned y = 0; y < th; y++)
      memcpy(packed + y * tw * 4, image + y * 64, tw * 4);
   image_descriptor(root + slot(0, 0), 0x900000, tw, th);
   sampler_descriptor(root + slot(0, 0) + 32);
   const float opacity[4] = {0.5f, 1.0f, 0.25f, 0.75f};
   for (unsigned c = 0; c < 4; c++)
      put(root + PUSH + 4 * c, bits(opacity[c]));
   uint32_t *attributes = calloc(APEX_ATTRIBUTE_BLOCK, sizeof(uint32_t));
   /* Input 0: uv (or a color); input 1: a flat color. */
   for (unsigned c = 0; c < 4; c++) {
      *attribute(attributes, c, 0) = bits(0.1f + 0.2f * c);
      *attribute(attributes, c, 1) = bits(0.5f - 0.125f * c);
      *attribute(attributes, c, 2) = bits(0.25f + 0.0625f * c);
      *attribute(attributes, 4 + c, 0) = bits(0.2f * (c + 1));
   }
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   fragment_wave(wave, attributes);
   run(&r, &w, wave->scalar, 1, wave);
   CHECK(wave->exported[0] == 0xffff);
   const uint8_t *descriptor = root + slot(0, 0);
   for (unsigned l = 0; l < 16; l++) {
      uint32_t i = wave->vector[0][l], j = wave->vector[1][l];
      /* A sample's coordinates are u and v: components 0 and 1. */
      for (unsigned c = 0; c < (textured ? 2 : 4); c++) {
         float want;
         if (textured) {
            want = flt(texel(descriptor, descriptor + 32, c, bits(interpolate(attributes, c, i, j)), FLAGS_2D));
            if (modulated)
               want = want * flt(*attribute(attributes, 4 + c, 0)) * opacity[c];
         } else {
            want = interpolate(attributes, c, i, j);
         }
         CHECK(wave->exports[0][l][c] == bits(want));
      }
   }
   (void)packed;
   free(wave);
   free(attributes);
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* mix.frag: mix(a, b, 1 - t) within rounding of a + (1 - t)(b - a). */
static void test_mix(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "mix.frag.spv", &r);
   struct world w = {0};
   region(&w, ROOT, ROOT_BYTES);
   uint32_t *attributes = calloc(APEX_ATTRIBUTE_BLOCK, sizeof(uint32_t));
   for (unsigned c = 0; c < 12; c++) {
      *attribute(attributes, c, 0) = bits(0.3f * c - 1.7f);
      *attribute(attributes, c, 1) = bits(0.45f - 0.07f * c);
      *attribute(attributes, c, 2) = bits(0.2f + 0.05f * c);
   }
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   fragment_wave(wave, attributes);
   run(&r, &w, wave->scalar, 1, wave);
   CHECK(wave->exported[0] == 0xffff);
   for (unsigned l = 0; l < 16; l++) {
      uint32_t i = wave->vector[0][l], j = wave->vector[1][l];
      for (unsigned c = 0; c < 4; c++) {
         double a = interpolate(attributes, c, i, j), b = interpolate(attributes, 4 + c, i, j);
         double t = interpolate(attributes, 8 + c, i, j), want = a + (1 - t) * (b - a);
         CHECK(fabs(flt(wave->exports[0][l][c]) - want) <= 4e-7 * fmax(fabs(a), fabs(b)));
      }
   }
   free(wave);
   free(attributes);
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* uniform.comp: the scalar FP forms return the vector forms' bits on normal,
 * signed-zero, subnormal, infinite and NaN operands. */
static void test_uniform(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "uniform.comp.spv", &r);
   unsigned seen = 0;
   for (size_t i = 64; i < r.size; i += 8)
      if (r.data[i] >= APEX_S_ADD_F && r.data[i] <= APEX_S_RSQ)
         seen |= 1u << (r.data[i] - APEX_S_ADD_F);
   CHECK(seen == 0x1f);
   static const float sets[][4] = {
      {1.5f, -2.25f, 0.75f, 0.4f}, {0.1f, 3.0f, 0.3f, 1.7f}, {-0.0f, 0.0f, -0.0f, -0.5f},
      {1e-39f, 0.5f, 3e-39f, 2.0f}, {INFINITY, 0.0f, 4.0f, NAN}, {3e38f, 2.0f, -3e38f, -INFINITY},
      {-4.0f, 0.25f, 16.0f, 0.999f},
   };
   const unsigned n = 9;
   for (unsigned k = 0; k < sizeof(sets) / sizeof(sets[0]); k++) {
      struct world w = {0};
      uint8_t *root = region(&w, ROOT, ROOT_BYTES);
      uint8_t *in = region(&w, 0x200000, 16 * 16);
      uint8_t *out = region(&w, 0x300000, 16 * 2 * n * 4);
      for (unsigned c = 0; c < 4; c++) {
         put(root + PUSH + 4 * c, bits(sets[k][c]));
         for (unsigned l = 0; l < 16; l++)
            put(in + 16 * l + 4 * c, bits(sets[k][c]));
      }
      buffer_descriptor(root + slot(0, 0), 0x200000, 16 * 16);
      buffer_descriptor(root + slot(0, 1), 0x300000, 16 * 2 * n * 4);
      uint32_t user[16];
      root_user(user);
      run(&r, &w, user, 1, NULL);
      for (unsigned l = 0; l < 16; l++)
         for (unsigned f = 0; f < n; f++)
            CHECK(get(out + 4 * (2 * n * l + f)) == get(out + 4 * (2 * n * l + n + f)));
      world_free(&w);
   }
   apex_compile_result_finish(&r);
}

/* hoist.frag with the branch taken (limit 3) and not (limit 0). */
static void test_hoist(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "hoist.frag.spv", &r);
   uint32_t *attributes = calloc(APEX_ATTRIBUTE_BLOCK, sizeof(uint32_t));
   for (unsigned c = 0; c < 3; c++) {
      *attribute(attributes, c, 0) = bits(0.4f * c - 0.3f);
      *attribute(attributes, c, 1) = bits(1.5f - 0.6f * c);
      *attribute(attributes, c, 2) = bits(0.7f + 0.2f * c);
   }
   for (unsigned taken = 0; taken < 2; taken++) {
      struct world w = {0};
      uint8_t *root = region(&w, ROOT, ROOT_BYTES);
      float limit = taken ? 3.0f : 0.0f;
      put(root + PUSH, bits(limit));
      struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
      fragment_wave(wave, attributes);
      run(&r, &w, wave->scalar, 1, wave);
      for (unsigned l = 0; l < 16; l++) {
         uint32_t i = wave->vector[0][l], j = wave->vector[1][l];
         double v[3], length = 0;
         for (unsigned c = 0; c < 3; c++) {
            v[c] = interpolate(attributes, c, i, j);
            length += v[c] * v[c];
         }
         length = sqrt(length);
         double attenuation = taken ? 1 - fmin(length / limit, 1) : 1;
         for (unsigned c = 0; c < 4; c++) {
            double want = c < 3 ? v[c] / length * attenuation : attenuation;
            CHECK(fabs(flt(wave->exports[0][l][c]) - want) <= 1e-5);
         }
      }
      free(wave);
      world_free(&w);
   }
   free(attributes);
   apex_compile_result_finish(&r);
}

/* Captured GNOME Shell and glmark2 (Zink) shaders: compile under the gate
 * and run once on the model against zeroed uniforms and a small texture in
 * every sampler slot. */
static void test_captured(const char *dir, unsigned expected)
{
   DIR *d = opendir(dir);
   CHECK(d);
   struct dirent *e;
   unsigned count = 0;
   while ((e = readdir(d))) {
      if (!strstr(e->d_name, ".spv"))
         continue;
      struct apex_compile_result r;
      compile(dir, e->d_name, &r);
      uint32_t stage = r.data[4];
      struct world w = {0};
      uint8_t *root = region(&w, ROOT, ROOT_BYTES);
      region(&w, 0xa00000, 65536);
      region(&w, 0x800000, 16 * 64 * 64);
      region(&w, 0x900000, 64 * 64);
      for (unsigned set = 0; set < 8; set++) {
         for (unsigned b = 0; b < 256; b++) {
            uint8_t *s = root + slot(set, b);
            if (set == 2) {
               image_descriptor(s, 0x900000, 4, 4);
               sampler_descriptor(s + 32);
            } else {
               buffer_descriptor(s, 0xa00000, 65536);
            }
         }
      }
      struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
      uint32_t *attributes = calloc(APEX_ATTRIBUTE_BLOCK, sizeof(uint32_t));
      if (stage == APEX_FRAGMENT) {
         fragment_wave(wave, attributes);
      } else {
         wave->exec = 0xffff;
         wave->scalar[0] = (uint32_t)ROOT;
         wave->scalar[16] = 0x800000;
      }
      run(&r, &w, wave->scalar, 1, wave);
      /* Every lane exports unless the program discards (header flag bit 1). */
      if (stage == APEX_FRAGMENT && !(r.data[5] & 2))
         CHECK(wave->exported[0] == 0xffff);
      free(wave);
      free(attributes);
      world_free(&w);
      apex_compile_result_finish(&r);
      count++;
   }
   closedir(d);
   CHECK(count == expected);
}

/* features.comp: two workgroups of four waves. */
static void test_features(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "features.comp.spv", &r);
   const unsigned groups = 2, limit = 40;
   struct world w = {0};
   uint8_t *root = region(&w, ROOT, ROOT_BYTES);
   uint8_t *out = region(&w, 0x200000, groups * 64 * 80);
   uint8_t *counter = region(&w, 0x300000, 17 * 4);
   buffer_descriptor(root + slot(0, 0), 0x200000, groups * 64 * 80);
   buffer_descriptor(root + slot(0, 1), 0x300000, 17 * 4);
   put(root + PUSH, limit);
   uint32_t user[16];
   root_user(user);
   user[2] = groups, user[3] = user[4] = 1;
   /* Private arrays need the private base in s20-s21 (header bytes per lane). */
   run_private(&r, &w, user, groups, NULL, 0x10000000);
   uint32_t hist[16] = {0};
   for (unsigned g = 0; g < groups * 64; g++) {
      uint32_t l = g % 64, wave = l / 16, lane = l % 16, group = g / 64;
      const uint8_t *o = out + g * 80;
      uint32_t acc = 0;
      for (uint32_t k = 0; k < limit; k++) {
         if ((k + l) % 3 == 0)
            continue;
         acc += k * (l + 1);
         if (acc > 500 + l * 7)
            break;
      }
      CHECK(get(o) == acc);
      uint32_t sum = 0, prefix = 0;
      for (unsigned k = 0; k < 16; k++) {
         sum += 16 * wave + k;
         if (k < lane)
            prefix += 16 * wave + k + 1;
      }
      CHECK(get(o + 4) == sum);
      CHECK(get(o + 8) == prefix);
      CHECK(get(o + 12) == 0x2222);
      CHECK(get(o + 16) == (16 * wave + ((lane + 5) & 15)) * 3);
      CHECK(get(o + 20) == (wave < 3));
      CHECK(get(o + 24) == 16 * wave);
      uint32_t a = 63 - l, b = (l + 17) & 63;
      CHECK(get(o + 28) == a * a + group + b * b + group);
      CHECK(get(o + 32) == 1);
      hist[l & 15] += l;
      CHECK(get(o + 36) == ((l * 5) & 7) * 11 + l + ((l + g) & 7) * 11 + l);
      uint64_t big = (uint64_t)(l + 1) * 0x100000001ull + 12345;
      CHECK(get(o + 40) == ((uint32_t)(big >> 17) ^ (uint32_t)(big >> 40)));
      CHECK(get(o + 44) == (l * 977 + 13) / (l % 7 + 1) + (l * 31) % 10);
      int32_t s = (int32_t)l - 32;
      int32_t modulo = ((s % 5) + 5) % 5; /* OpSMod takes the divisor's sign */
      CHECK(get(o + 48) == (uint32_t)(s / 5) + (uint32_t)modulo * 100);
      float x = (float)l * 0.25f - 5.0f;
      float f = floorf(x) + fabsf(x) * 0.5f + fminf(x, 1.0f) + fmaxf(x, -2.0f);
      CHECK(get(o + 52) == bits(f));
      float cl = x < -1.0f ? -1.0f : x > 1.0f ? 1.0f : x;
      CHECK(get(o + 56) == (uint32_t)(cl * 1000.0f + 5000.0f));
      uint32_t v = l * 1234 + 1, msb = 31 - __builtin_clz(v);
      uint32_t rev = 0;
      for (unsigned k = 0; k < 32; k++)
         rev |= ((l >> k) & 1) << (31 - k);
      CHECK(get(o + 60) == (msb | __builtin_popcount(l * 77) << 8 |
                            (uint32_t)__builtin_ctz(l + 64) << 16 | (rev >> 24) << 24));
      CHECK(get(o + 64) == ((l ^ 1) | (l ^ 3) << 8));
      CHECK(get(o + 68) == ((l & ~3u) + 2) * 7);
      unsigned below = 0;
      for (unsigned k = 16 * wave; k < l; k++)
         below += k % 3 == 0;
      CHECK(get(o + 72) == below);
      CHECK(get(o + 76) == 4 * (l & ~3u) + 6);
   }
   CHECK(get(counter) == groups * 64);
   for (unsigned k = 0; k < 16; k++)
      CHECK(get(counter + 4 + 4 * k) == hist[k]);
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* spill.comp: values that exceed the register file round-trip through private memory. */
static void test_spill(const char *dir)
{
   struct apex_compile_result r;
   compile_spilling(dir, "spill.comp.spv", &r, true);
   struct world w = {0};
   uint8_t *root = region(&w, ROOT, ROOT_BYTES);
   const unsigned words = 16 * 34 * 4;
   uint8_t *src = region(&w, 0x200000, words * 4);
   uint8_t *dst = region(&w, 0x400000, words * 4);
   for (unsigned i = 0; i < words; i++)
      put(src + 4 * i, 0x2545f491u * (i + 7));
   buffer_descriptor(root + slot(0, 0), 0x200000, words * 4);
   buffer_descriptor(root + slot(0, 1), 0x400000, words * 4);
   put(root + PUSH, 1);
   uint32_t user[16];
   root_user(user);
   run_private(&r, &w, user, 1, NULL, 0x10000000);
   for (unsigned l = 0; l < 16; l++) {
      for (unsigned k = 0; k < 17; k++) {
         for (unsigned c = 0; c < 4; c++) {
            uint32_t b = get(src + 16 * (l * 34 + 17 + 16 - k) + 4 * c);
            uint32_t a = get(src + 16 * (l * 34 + 16 - k) + 4 * c);
            CHECK(get(dst + 16 * (l * 34 + k) + 4 * c) == b + k);
            CHECK(get(dst + 16 * (l * 34 + 17 + k) + 4 * c) == (a ^ l));
         }
      }
   }
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* waterfall.comp: three buffers selected per lane. */
static void test_waterfall(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "waterfall.comp.spv", &r);
   struct world w = {0};
   uint8_t *root = region(&w, ROOT, ROOT_BYTES);
   uint8_t *src = region(&w, 0x200000, 3 * 256);
   uint8_t *dst = region(&w, 0x300000, 64);
   for (unsigned b = 0; b < 3; b++) {
      for (unsigned k = 0; k < 16; k++)
         put(src + 256 * b + 4 * k, 1000 * (b + 1) + k);
      buffer_descriptor(root + slot(0, b), 0x200000 + 256 * b, 64 + 16 * b);
   }
   buffer_descriptor(root + slot(0, 3), 0x300000, 64);
   uint32_t user[16];
   root_user(user);
   run(&r, &w, user, 1, NULL);
   for (unsigned l = 0; l < 16; l++) {
      unsigned b = l % 3;
      CHECK(get(dst + 4 * l) == 1000 * (b + 1) + l + (64 + 16 * b) / 4);
   }
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* control.comp: host replica of the nested control flow. */
static void control_reference(uint32_t l, uint32_t n, uint32_t mode, const uint32_t *v, uint32_t out[3])
{
   uint32_t a = 0, b = 7;
   for (uint32_t i = 0; i < n + (l & 3); i++) {
      if (((i ^ l) & 1) == 0) {
         a += v[(i + l) & 63];
         if (mode == 1)
            b = b * 3 + i;
         else
            b ^= a;
      } else {
         b += i;
         for (uint32_t j = 0; j < (l & 3); j++) {
            a += j * b;
            if (a > 100000)
               break;
         }
      }
   }
   out[0] = a;
   out[1] = b;
   out[2] = l < 20 ? a + b : l < 25 ? a * b : v[l];
}
static void test_control(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "control.comp.spv", &r);
   for (uint32_t mode = 0; mode < 2; mode++) {
      struct world w = {0};
      uint8_t *root = region(&w, ROOT, ROOT_BYTES);
      uint8_t *out = region(&w, 0x200000, 32 * 16);
      uint8_t *in = region(&w, 0x300000, 256);
      uint32_t v[64];
      for (unsigned k = 0; k < 64; k++) {
         v[k] = 0x01000193u * (k + 3) >> 8;
         put(in + 4 * k, v[k]);
      }
      buffer_descriptor(root + slot(0, 0), 0x200000, 32 * 16);
      buffer_descriptor(root + slot(0, 1), 0x300000, 256);
      put(root + PUSH, 9);
      put(root + PUSH + 4, mode);
      uint32_t user[16];
      root_user(user);
      run(&r, &w, user, 1, NULL);
      for (uint32_t l = 0; l < 32; l++) {
         uint32_t want[3];
         control_reference(l, 9, mode, v, want);
         for (unsigned k = 0; k < 3; k++)
            CHECK(get(out + 16 * l + 4 * k) == want[k]);
         CHECK(get(out + 16 * l + 12) == (l % 5) * 9 + 5);
      }
      world_free(&w);
   }
   apex_compile_result_finish(&r);
}

/* texture.comp: an 8x4 RGBA8 image with a nearest, repeating sampler. */
static void test_texture(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "texture.comp.spv", &r);
   struct world w = {0};
   uint8_t *root = region(&w, ROOT, ROOT_BYTES);
   uint8_t *image = region(&w, 0x900000, 64 * 4);
   uint8_t *out = region(&w, 0x200000, 16 * 80);
   uint8_t packed[8 * 4 * 4];
   for (unsigned k = 0; k < sizeof(packed); k++) {
      packed[k] = (uint8_t)(k * 29 + 5);
      image[(k / 32) * 64 + k % 32] = packed[k];
   }
   image_descriptor(root + slot(0, 0), 0x900000, 8, 4);
   sampler_descriptor(root + slot(0, 0) + 32);
   buffer_descriptor(root + slot(0, 1), 0x200000, 16 * 80);
   uint32_t user[16];
   root_user(user);
   run(&r, &w, user, 1, NULL);
   const uint8_t *d = root + slot(0, 0), *s = d + 32;
   const uint32_t fetch = 4u << 13 | FLAGS_2D, level = 1u << 13 | FLAGS_2D, gather = 5u << 13 | FLAGS_2D | 1u << 19;
   const uint32_t offset = level | (-3 & 15) | 7 << 4;
   for (unsigned l = 0; l < 16; l++) {
      unsigned x = l % 8, y = l / 8;
      const uint8_t *o = out + 80 * l;
      float u = (x + 0.5f) / 8.0f, v = (y + 0.5f) / 4.0f;
      /* Fetch: integer x, y and level 0 in a+3; the image descriptor is the sampler. */
      CHECK(get(o) == texel(d, d, 0, x, fetch) && get(o + 4) == texel(d, d, 1, y, fetch) &&
            get(o + 12) == texel(d, d, 3, 0, fetch));
      /* Level: u, v and the level 0.0 in a+3. */
      CHECK(get(o + 16) == texel(d, s, 0, bits(u + 0.125f), level) && get(o + 20) == texel(d, s, 1, bits(v), level) &&
            get(o + 28) == texel(d, s, 3, 0, level));
      /* Gather: u, v. */
      CHECK(get(o + 32) == texel(d, s, 0, bits(u), gather) && get(o + 36) == texel(d, s, 1, bits(v), gather));
      CHECK(get(o + 48) == bits(8.0f) && get(o + 52) == bits(4.0f) && get(o + 56) == bits(1.0f));
      /* Offsets travel in the instruction, not in coordinate registers. */
      CHECK(get(o + 64) == texel(d, s, 0, bits(u), offset) && get(o + 76) == texel(d, s, 3, 0, offset));
   }
   (void)packed;
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* discard.frag and demote.frag: quads 2-3 have v.x above 0.45. */
static void test_kill(const char *dir, const char *name, bool demote)
{
   struct apex_compile_result r;
   compile(dir, name, &r);
   struct world w = {0};
   region(&w, ROOT, ROOT_BYTES);
   uint32_t *attributes = calloc(APEX_ATTRIBUTE_BLOCK, sizeof(uint32_t));
   *attribute(attributes, 0, 1) = bits(1.0f); /* x = i */
   *attribute(attributes, 1, 0) = bits(0.5f), *attribute(attributes, 1, 1) = bits(3.0f);
   *attribute(attributes, 1, 2) = bits(-2.0f);
   *attribute(attributes, 2, 0) = bits(0.75f);
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   fragment_wave(wave, attributes);
   run(&r, &w, wave->scalar, 1, wave);
   uint16_t survivors = 0;
   for (unsigned l = 0; l < 16; l++) {
      float x = interpolate(attributes, 0, wave->vector[0][l], wave->vector[1][l]);
      if (demote ? !(x > 0.45f && x < 0.55f) : !(x > 0.45f))
         survivors |= 1u << l;
   }
   CHECK(survivors != 0xffff && survivors);
   CHECK(wave->exported[0] == survivors);
   for (unsigned l = 0; l < 16; l++) {
      if (!(survivors >> l & 1))
         continue;
      unsigned q = l & ~3u;
      float y0 = interpolate(attributes, 1, wave->vector[0][q], wave->vector[1][q]);
      float y1 = interpolate(attributes, 1, wave->vector[0][q + 1], wave->vector[1][q + 1]);
      float y2 = interpolate(attributes, 1, wave->vector[0][q + 2], wave->vector[1][q + 2]);
      CHECK(wave->exports[0][l][0] == bits(y1 - y0));
      CHECK(wave->exports[0][l][1] == bits(demote ? 0.0f : y2 - y0));
      CHECK(wave->exports[0][l][2] == bits(0.75f));
      CHECK(wave->exports[0][l][3] == bits(1.0f));
   }
   free(wave);
   free(attributes);
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* depth.frag: the depth plane at the pixel centre from its origin, and
 * noperspective weights i Q / q1, j Q / q2 (q1 = q2 = 1, Q = v4). */
static void test_depth(const char *dir)
{
   struct apex_compile_result r;
   compile(dir, "depth.frag.spv", &r);
   CHECK(r.data[5] & 1u << 6);
   struct world w = {0};
   region(&w, ROOT, ROOT_BYTES);
   uint32_t *attributes = calloc(APEX_ATTRIBUTE_BLOCK, sizeof(uint32_t));
   attributes[0] = bits(0.25f), attributes[1] = attributes[2] = bits(1.0f);
   attributes[4] = bits(0.5f), attributes[5] = bits(0.0625f), attributes[6] = bits(-0.03125f);
   attributes[7] = 3 | 1 << 16; /* origin (3, 1) */
   for (unsigned c = 0; c < 2; c++) {
      *attribute(attributes, c, 0) = bits(0.125f + c);
      *attribute(attributes, c, 1) = bits(2.0f);
      *attribute(attributes, c, 2) = bits(-1.0f - c);
   }
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   fragment_wave(wave, attributes);
   for (unsigned l = 0; l < 16; l++) {
      wave->vector[2][l] = (8 + 2 * (l >> 2) + (l & 1)) | (4 + ((l >> 1) & 1)) << 16;
      wave->vector[4][l] = bits(0.5f + 0.125f * (l & 3));
   }
   run(&r, &w, wave->scalar, 1, wave);
   CHECK(wave->exported[0] == 0xffff);
   for (unsigned l = 0; l < 16; l++) {
      uint32_t p = wave->vector[2][l];
      float fx = (float)((int)(p & 0xffff) - 3) + 0.5f, fy = (float)((int)(p >> 16) - 1) + 0.5f;
      float z = fmaf(fy, -0.03125f, fmaf(fx, 0.0625f, 0.5f));
      float q = flt(wave->vector[4][l]);
      uint32_t i = bits(flt(wave->vector[0][l]) * q), j = bits(flt(wave->vector[1][l]) * q);
      CHECK(wave->exports[0][l][0] == bits(z));
      CHECK(wave->exports[0][l][1] == bits(interpolate(attributes, 0, i, j)));
      CHECK(wave->exports[0][l][2] == bits(q));
      CHECK(wave->exports[0][l][3] == bits(interpolate(attributes, 1, i, j)));
   }
   free(wave);
   free(attributes);
   world_free(&w);
   apex_compile_result_finish(&r);
}

int main(int argc, char **argv)
{
   CHECK(argc == 4);
   setvbuf(stdout, NULL, _IOLBF, 0);
   test_bench_compute(argv[1]);
   test_vertex(argv[1], "bench.vert.spv", 2);
   test_vertex(argv[1], "triangle.vert.spv", 4);
   test_fragment(argv[1], "bench.frag.spv", true, false);
   test_fragment(argv[1], "triangle.frag.spv", false, false);
   test_fragment(argv[1], "compositor.frag.spv", true, true);
   test_transform(argv[1]);
   test_mix(argv[1]);
   test_hoist(argv[1]);
   test_uniform(argv[1]);
   test_features(argv[1]);
   test_spill(argv[1]);
   test_waterfall(argv[1]);
   test_control(argv[1]);
   test_texture(argv[1]);
   test_kill(argv[1], "discard.frag.spv", false);
   test_kill(argv[1], "demote.frag.spv", true);
   test_depth(argv[1]);
   test_captured(argv[2], 24);
   test_captured(argv[3], 54);
   printf("PASS Apex shaders: %u programs, %u instructions, max s%u v%u, spills only in spill.comp\n", shaders,
          total_instructions, max_scalar, max_vector);
   return 0;
}
