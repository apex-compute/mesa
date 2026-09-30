#version 450
// Offline compile fixture: clip and cull distances in the vertex kernel.
layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 shade;
out gl_PerVertex {
   vec4 gl_Position;
   float gl_ClipDistance[5];
   float gl_CullDistance[1];
};
void main()
{
   gl_Position = position;
   for (int i = 0; i < 5; i++)
      gl_ClipDistance[i] = position[i & 3] + float(i);
   gl_CullDistance[0] = position.z;
   shade = color;
}
