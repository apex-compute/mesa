#version 450
// Hidden attributes: window depth from the depth plane, noperspective
// weights from the vertices' 1/w, and 1/w itself.
layout(location = 0) noperspective in vec4 v;
layout(location = 0) out vec4 target;
void main()
{
   target = vec4(gl_FragCoord.z, v.x, gl_FragCoord.w, v.y);
}
