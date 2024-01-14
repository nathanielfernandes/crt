#ifndef CAMERAH
#define CAMERAH

#include "ray.cu"

class camera {
public:
  __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov,
                    float aspect, float aperture, float focus_dist) {
    // vfov is top to bottom in degrees
    vec3 u, v, w;

    float theta = vfov * M_PI / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;

    origin = lookfrom;

    w = normalize(lookfrom - lookat);
    u = normalize(cross(vup, w));
    v = cross(w, u);

    llc = origin - half_width * focus_dist * u - half_height * focus_dist * v -
          focus_dist * w;

    horizontal = 2.0f * half_width * focus_dist * u;
    vertical = 2.0f * half_height * focus_dist * v;

    lens_radius = aperture / 2.0f;
  }

  __device__ ray get_ray(float u, float v) {
    return ray(origin, llc + u * horizontal + v * vertical - origin);
  }

  vec3 origin, llc, horizontal, vertical;
  float lens_radius;
};

#endif