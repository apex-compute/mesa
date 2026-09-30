#version 450
// Window composition: a texture modulated by a flat vertex color and a
// push-constant opacity (the texture-times-color bypass forms).
layout(set = 0, binding = 0) uniform sampler2D image;
layout(location = 0) in vec2 uv;
layout(location = 1) flat in vec4 color;
layout(push_constant) uniform Parameters { vec4 opacity; } p;
layout(location = 0) out vec4 target;
void main()
{
   target = texture(image, uv) * color * p.opacity;
}
