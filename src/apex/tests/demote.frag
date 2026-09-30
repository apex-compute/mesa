#version 450
// Demoted lanes keep running as helpers for derivatives but do not export.
#extension GL_EXT_demote_to_helper_invocation : require
layout(location = 0) in vec4 v;
layout(location = 0) out vec4 target;
void main()
{
   if (v.x > 0.45 && v.x < 0.55)
      demote;
   target = vec4(dFdx(v.y), helperInvocationEXT() ? 1.0 : 0.0, v.z, 1.0);
}
