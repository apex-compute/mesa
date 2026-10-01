#version 450
// Matrix columns summed term by term, as Zink's GLSL lowering emits them:
// the varying takes its translation into the first product, the position
// keeps the order its expression gives.
layout(push_constant) uniform P { mat4 m, n; } p;
layout(location = 0) in vec4 v;
layout(location = 0) out vec4 world;
void main()
{
   gl_Position = p.m[0] * v.x + p.m[1] * v.y + p.m[2] * v.z + p.m[3];
   world = p.n[0] * v.x + p.n[1] * v.y + p.n[2] * v.z + p.n[3];
}
