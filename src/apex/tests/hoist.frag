#version 450
// The branch's length(l) and the normalize after it share one dot product.
layout(push_constant) uniform P { float limit; } p;
layout(location = 0) in vec3 l;
layout(location = 0) out vec4 target;
void main()
{
   float attenuation = 1.0;
   if (p.limit > 0.0)
      attenuation = 1.0 - min(length(l) / p.limit, 1.0);
   target = vec4(normalize(l) * attenuation, attenuation);
}
