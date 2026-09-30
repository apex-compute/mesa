/* SPDX-License-Identifier: MIT */
/* Compiles the benchmark, triangle, compositor and captured GNOME Shell
 * shaders, runs them on the ISA model against host references, and holds the
 * zero-spill gate. Usage: apex-shader-test DIR GNOME_DIR where DIR holds the
 * SPIR-V of bench.{comp,vert,frag}, triangle.{vert,frag} and compositor.frag. */
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
/* Linear RGBA8 2D image and a nearest, repeating sampler (isa-notes.md). */
static void image_descriptor(uint8_t *p, uint64_t va, unsigned w, unsigned h)
{
   put(p, (uint32_t)va);
   put(p + 4, (uint32_t)(va >> 32) | 1u << 8 | 1u << 17);
   put(p + 8, (w - 1) | (h - 1) << 14);
   put(p + 12, 0);
   put(p + 20, (w * 4 + 63) / 64);
}
static void sampler_descriptor(uint8_t *p)
{
   put(p, 0);
   put(p + 4, 0);
   put(p + 8, 0);
}

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

static uint64_t run(const struct apex_compile_result *r, struct world *w, const uint32_t user[16],
                    unsigned groups, struct apex_sim_wave *wave)
{
   uint32_t grid[3] = {groups, 1, 1};
   uint64_t executed = 0;
   char diagnostic[256] = "";
   if (apex_simulate(r->data, r->size, user, grid, 0, wave, w->r, w->n, &executed, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
   return executed;
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
      uint64_t executed = run(&r, &w, user, groups, NULL);
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
      printf("  bench.comp mode %u: %llu wave instructions\n", mode, (unsigned long long)executed);
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
 * then varying L at record vec4 2 + L. */
static void test_vertex(const char *dir, const char *name, unsigned varying_components)
{
   struct apex_compile_result r;
   compile(dir, name, &r);
   uint32_t stride;
   memcpy(&stride, r.data + 28, 4);
   CHECK(stride == 48);
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
         CHECK(get(out + l * stride + 32 + 4 * c) == wave->vector[6 + c][l]);
   }
   free(wave);
   world_free(&w);
   apex_compile_result_finish(&r);
}

static float texel(const uint8_t *image, unsigned width, unsigned height, float u, float v,
                   unsigned c)
{
   int x = (int)floorf(u * width), y = (int)floorf(v * height);
   x = ((x % (int)width) + width) % width;
   y = ((y % (int)height) + height) % height;
   return image[(y * width + x) * 4 + c] / 255.0f;
}

/* Fragment waves: i and j per lane in v0-v1; attribute block of primitive 0. */
static void fragment_wave(struct apex_sim_wave *wave, float *attributes)
{
   wave->exec = 0xffff;
   wave->scalar[0] = (uint32_t)ROOT;
   for (unsigned l = 0; l < 16; l++) {
      wave->vector[0][l] = bits(0.03125f * (l & 3) + 0.25f * (l >> 2));
      wave->vector[1][l] = bits(0.0625f * (l >> 2));
      wave->vector[2][l] = (l & 1) | (l >> 1) << 16;
   }
   wave->attributes = attributes;
   wave->primitives = 1;
}
static float interpolate(const float *attributes, unsigned component, uint32_t i, uint32_t j)
{
   const float *p = attributes + 3 * component;
   return fmaf(flt(j), p[2], fmaf(flt(i), p[1], p[0]));
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
   float *attributes = calloc(144 * 3, sizeof(float));
   /* Input 0: uv (or a color); input 1: a flat color. */
   for (unsigned c = 0; c < 4; c++) {
      attributes[3 * c] = 0.1f + 0.2f * c;
      attributes[3 * c + 1] = 0.5f - 0.125f * c;
      attributes[3 * c + 2] = 0.25f + 0.0625f * c;
      attributes[3 * (4 + c)] = 0.2f * (c + 1);
   }
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   fragment_wave(wave, attributes);
   run(&r, &w, wave->scalar, 1, wave);
   CHECK(wave->exported[0] == 0xffff);
   for (unsigned l = 0; l < 16; l++) {
      uint32_t i = wave->vector[0][l], j = wave->vector[1][l];
      for (unsigned c = 0; c < 4; c++) {
         float want;
         if (textured) {
            float u = interpolate(attributes, 0, i, j), v = interpolate(attributes, 1, i, j);
            want = texel(packed, tw, th, u, v, c);
            if (modulated)
               want = want * attributes[3 * (4 + c)] * opacity[c];
         } else {
            want = interpolate(attributes, c, i, j);
         }
         CHECK(wave->exports[0][l][c] == bits(want));
      }
   }
   free(wave);
   free(attributes);
   world_free(&w);
   apex_compile_result_finish(&r);
}

/* GNOME Shell (Zink) shaders: compile under the gate and run once on the model
 * against zeroed uniforms and a small texture in every sampler slot. */
static void test_gnome(const char *dir)
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
      uint32_t stage;
      memcpy(&stage, r.data + 4, 4);
      stage &= 0xff;
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
      float *attributes = calloc(144 * 3, sizeof(float));
      if (stage == APEX_FRAGMENT) {
         fragment_wave(wave, attributes);
      } else {
         wave->exec = 0xffff;
         wave->scalar[0] = (uint32_t)ROOT;
         wave->scalar[16] = 0x800000;
      }
      run(&r, &w, wave->scalar, 1, wave);
      if (stage == APEX_FRAGMENT)
         CHECK(wave->exported[0] == 0xffff);
      free(wave);
      free(attributes);
      world_free(&w);
      apex_compile_result_finish(&r);
      count++;
   }
   closedir(d);
   CHECK(count >= 24);
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
   uint32_t private_bytes;
   memcpy(&private_bytes, r.data + 24, 4);
   uint8_t *private = region(&w, 0x10000000, 4 * 16 * private_bytes + 64);
   uint32_t grid[3] = {groups, 1, 1};
   uint64_t executed = 0;
   char diagnostic[256] = "";
   if (apex_simulate(r.data, r.size, user, grid, 0x10000000, NULL, w.r, w.n, &executed, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
   (void)private;
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
   printf("  features.comp: %llu wave instructions\n", (unsigned long long)executed);
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
   uint32_t private_bytes;
   memcpy(&private_bytes, r.data + 24, 4);
   region(&w, 0x10000000, 16 * private_bytes);
   uint32_t user[16];
   root_user(user);
   uint32_t grid[3] = {1, 1, 1};
   uint64_t executed = 0;
   char diagnostic[256] = "";
   CHECK(!apex_simulate(r.data, r.size, user, grid, 0x10000000, NULL, w.r, w.n, &executed, diagnostic));
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

/* discard.frag and demote.frag: quads 2-3 have v.x above 0.45. */
static void test_kill(const char *dir, const char *name, bool demote)
{
   struct apex_compile_result r;
   compile(dir, name, &r);
   struct world w = {0};
   region(&w, ROOT, ROOT_BYTES);
   float *attributes = calloc(144 * 3, sizeof(float));
   attributes[3 * 0 + 1] = 1.0f;                  /* x = i */
   attributes[3 * 1] = 0.5f, attributes[3 * 1 + 1] = 3.0f, attributes[3 * 1 + 2] = -2.0f;
   attributes[3 * 2] = 0.75f;
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

int main(int argc, char **argv)
{
   CHECK(argc == 3);
   setvbuf(stdout, NULL, _IOLBF, 0);
   test_bench_compute(argv[1]);
   test_vertex(argv[1], "bench.vert.spv", 2);
   test_vertex(argv[1], "triangle.vert.spv", 4);
   test_fragment(argv[1], "bench.frag.spv", true, false);
   test_fragment(argv[1], "triangle.frag.spv", false, false);
   test_fragment(argv[1], "compositor.frag.spv", true, true);
   test_features(argv[1]);
   test_spill(argv[1]);
   test_kill(argv[1], "discard.frag.spv", false);
   test_kill(argv[1], "demote.frag.spv", true);
   test_gnome(argv[2]);
   printf("PASS Apex shaders: %u programs, %u instructions, max s%u v%u, spills only in spill.comp\n", shaders,
          total_instructions, max_scalar, max_vector);
   return 0;
}
