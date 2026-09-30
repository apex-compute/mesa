#version 450
// Offline compile fixture: transform feedback outputs in two buffers.
layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(location = 0, xfb_buffer = 0, xfb_offset = 0) out vec4 shade;
layout(location = 1, xfb_buffer = 1, xfb_offset = 4) out vec2 pair;
out gl_PerVertex {
   layout(xfb_buffer = 0, xfb_offset = 16) vec4 gl_Position;
};
layout(xfb_buffer = 0, xfb_stride = 32) out;
layout(xfb_buffer = 1, xfb_stride = 16) out;
void main()
{
   gl_Position = position;
   shade = color;
   pair = position.xy;
}
