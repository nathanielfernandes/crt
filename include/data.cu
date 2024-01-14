#ifndef DATA_H
#define DATA_H

#include "vecs.cu"

//    vertex/normal
//       ___|___
// vec4 (x, y, z, w)
//                |
//           uv texture coords
typedef vec4 VertexU;
typedef vec4 NormalV;

// struct represents indexes into a vertex and normal buffer
// material is an index into a material buffer
struct Triangle {
  int v0, v1, v2;
  int material;
};

enum MaterialType { LAMBERTIAN = 0, METAL = 1, DIELECTRIC = 2, EMISSIVE = 3 };

struct Material {
  MaterialType type;
  float p1; // roughness for metal, refraction index for dielectric
};

#endif
