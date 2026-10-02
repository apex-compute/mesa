// Surface addressing of the internal kernels (apex_job.h): texel (x, y) of
// plane z = layer * samples + sample, relative to the surface's origin. Tiled
// surfaces follow the texture unit's layout (apex_hw.h): a level is a
// row-major grid of 4 KiB tiles of 8 x 8 Morton-ordered 64-byte blocks, a
// block holds texels in 4 x 4 sub-blocks, left to right then top to bottom,
// row-major within each. The including kernel declares `uint job(uint)`.


uint64_t surface_va(uint s)
{
   return uint64_t(job(s + APEX_SURFACE_VA)) | (uint64_t(job(s + APEX_SURFACE_VA + 1u)) << 32);
}

uint surface_texel_bytes(uint s)
{
   return job(s + APEX_SURFACE_TEXEL);
}

uint64_t surface_address(uint s, uint x, uint y, uint z)
{
   uint flags = job(s + APEX_SURFACE_FLAGS);
   uint samples = max(flags >> 4 & 255u, 1u);
   uint layer = z / samples, plane = z - layer * samples;
   uint origin = job(s + APEX_SURFACE_ORIGIN);
   x += origin & 0xffffu;
   y += origin >> 16;
   uint64_t base = surface_va(s) + uint64_t(layer) * job(s + APEX_SURFACE_LAYER) +
                   uint64_t(plane) * job(s + APEX_SURFACE_SAMPLE);
   if ((flags & APEX_SURFACE_TILED) == 0u)
      return base + uint64_t(y) * job(s + APEX_SURFACE_PITCH) + uint64_t(x) * surface_texel_bytes(s);
   // Block width and height logs by log2 texel bytes: 3 3 2 2 2, 3 2 2 1 0.
   uint lb = flags >> 1 & 7u, bw = lb < 2u ? 3u : 2u, bh = lb == 0u ? 3u : lb < 3u ? 2u : 4u - lb;
   uint page = (y >> (bh + 3u)) * job(s + APEX_SURFACE_PITCH) + (x >> (bw + 3u));
   uint bx = (x >> bw) & 7u, by = (y >> bh) & 7u;
   uint block = 0u;
   for (uint i = 0u; i < 3u; i++)
      block |= ((bx >> i) & 1u) << (2u * i) | ((by >> i) & 1u) << (2u * i + 1u);
   uint sh = min(bh, 2u), tx = x & ((1u << bw) - 1u), ty = y & ((1u << bh) - 1u);
   uint within = (((ty >> 2) << (bw - 2u)) + (tx >> 2)) * (4u << sh) + (ty & ((1u << sh) - 1u)) * 4u +
                 (tx & 3u);
   return base + uint64_t(page) * 4096u + block * 64u + (within << lb);
}
