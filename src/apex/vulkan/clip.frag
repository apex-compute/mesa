#version 450
// Offline compile fixture: clip and cull distances as fragment inputs.
layout(location = 0) in vec4 shade;
layout(location = 0) out vec4 target;
in float gl_ClipDistance[5];
in float gl_CullDistance[1];
void main()
{
   int i = int(shade.x * 4.0);
   target = vec4(gl_ClipDistance[i], gl_ClipDistance[4], gl_CullDistance[0], shade.w);
}
