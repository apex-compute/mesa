#version 450
// Terminated quads leave the export; survivors take coarse derivatives.
layout(location = 0) in vec4 v;
layout(location = 0) out vec4 target;
void main()
{
   if (v.x > 0.45)
      discard;
   target = vec4(dFdx(v.y), dFdy(v.y), v.z, 1.0);
}
