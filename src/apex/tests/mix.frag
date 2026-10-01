#version 450
layout(location = 0) in vec4 a;
layout(location = 1) in vec4 b;
layout(location = 2) in vec4 t;
layout(location = 0) out vec4 target;
void main()
{
   target = mix(a, b, 1.0 - t);
}
