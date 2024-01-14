#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.cu"

struct HitRecord {
  vec3 point, normal, ffnormal, bary;
  float u, v, dist;
};

#endif