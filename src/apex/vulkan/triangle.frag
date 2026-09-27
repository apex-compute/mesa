#version 450
layout(location = 0) in vec4 shade;
layout(location = 0) out vec4 target;
void main()
{
   target = shade;
}
