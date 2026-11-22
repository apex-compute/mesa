#version 450
// 136 dwords stay live across a uniform branch: a vertex program on a device
// with graphics private memory spills them, and the results must survive the
// round trip. Without it the program does not compile.
layout(std430, set = 0, binding = 0) readonly buffer Source { uvec4 src[]; };
layout(std430, set = 0, binding = 1) buffer Destination { uvec4 dst[]; };
layout(push_constant) uniform Parameters { uint enable; } p;
void main()
{
   uint l = uint(gl_VertexIndex);
   uvec4 a[17], b[17];
   for (uint k = 0u; k < 17u; k++)
      a[k] = src[l * 34u + k];
   for (uint k = 0u; k < 17u; k++)
      b[k] = src[l * 34u + 17u + k];
   if (p.enable != 0u) {
      for (uint k = 0u; k < 17u; k++)
         dst[l * 34u + k] = b[16u - k] + uvec4(k);
      for (uint k = 0u; k < 17u; k++)
         dst[l * 34u + 17u + k] = a[16u - k] ^ uvec4(l);
   }
   gl_Position = vec4(0.0);
}
