#version 450
// Every attribute through an indexed input array in a loop, as the CTS
// vertex_input max_attributes cases read them. The loop unrolls and each
// attribute is fetched where it is read, so the 31 vec4s are never all
// live: as select trees in the loop they overflowed the register file.
layout(location = 1) in vec4 attr[31];
void main()
{
   int ok = 0;
   for (int i = 1; i < 32; i++) {
      float index = float(i) + 0.25 * float(gl_VertexIndex);
      if (abs(attr[i - 1][0] - index) < 0.01)
         ok++;
      if (abs(attr[i - 1][1] - (index + 1.0)) < 0.01)
         ok++;
      if (abs(attr[i - 1][2] - (index + 2.0)) < 0.01)
         ok++;
      if (abs(attr[i - 1][3] - (index + 3.0)) < 0.01)
         ok++;
   }
   gl_Position = vec4(float(ok), 0.0, 0.0, 1.0);
}
