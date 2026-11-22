#version 450
// spill.vert's round trip in a fragment program; lane index 2x + y of the
// pixel.
layout(std430, set = 0, binding = 0) readonly buffer Source { uvec4 src[]; };
layout(std430, set = 0, binding = 1) buffer Destination { uvec4 dst[]; };
layout(push_constant) uniform Parameters { uint enable; } p;
layout(location = 0) out vec4 color;
void main()
{
   uint l = 2u * uint(gl_FragCoord.x) + uint(gl_FragCoord.y);
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
   color = vec4(1.0);
}
