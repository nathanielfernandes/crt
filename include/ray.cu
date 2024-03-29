#ifndef RAY_H
#define RAY_H

#include "vecs.cu"

class ray {
public:
  __device__ ray() {}
  __device__ ray(const vec3 &orig, const vec3 &dir) : orig(orig), dir(dir) {}

  __device__ vec3 origin() const { return orig; }
  __device__ vec3 direction() const { return dir; }
  __device__ vec3 point_at(float t) const { return orig + t * dir; }

  vec3 orig;
  vec3 dir;
};

#endif