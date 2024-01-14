#ifndef VEC3H
#define VEC3H

#include <iostream>
#include <math.h>

class vec3 {

public:
  __host__ __device__ vec3() {}
  __host__ __device__ vec3(float x, float y, float z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }
  __host__ __device__ inline float x() const { return data[0]; }
  __host__ __device__ inline float y() const { return data[1]; }
  __host__ __device__ inline float z() const { return data[2]; }
  __host__ __device__ inline float r() const { return data[0]; }
  __host__ __device__ inline float g() const { return data[1]; }
  __host__ __device__ inline float b() const { return data[2]; }

  __host__ __device__ inline const vec3 &operator+() const { return *this; }
  __host__ __device__ inline vec3 operator-() const {
    return vec3(-data[0], -data[1], -data[2]);
  }
  __host__ __device__ inline float operator[](int i) const { return data[i]; }
  __host__ __device__ inline float &operator[](int i) { return data[i]; };

  __host__ __device__ inline vec3 &operator+=(const vec3 &v2);
  __host__ __device__ inline vec3 &operator-=(const vec3 &v2);
  __host__ __device__ inline vec3 &operator*=(const vec3 &v2);
  __host__ __device__ inline vec3 &operator/=(const vec3 &v2);
  __host__ __device__ inline vec3 &operator*=(const float t);
  __host__ __device__ inline vec3 &operator/=(const float t);

  __host__ __device__ inline float length() const {
    return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }
  __host__ __device__ inline float squared_length() const {
    return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
  }
  __host__ __device__ inline void make_unit_vector();

  float data[3];
};

inline std::istream &operator>>(std::istream &is, vec3 &t) {
  is >> t.data[0] >> t.data[1] >> t.data[2];
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec3 &t) {
  os << t.data[0] << " " << t.data[1] << " " << t.data[2];
  return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
  float k =
      1.0 / sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  data[0] *= k;
  data[1] *= k;
  data[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.data[0] + v2.data[0], v1.data[1] + v2.data[1],
              v1.data[2] + v2.data[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.data[0] - v2.data[0], v1.data[1] - v2.data[1],
              v1.data[2] - v2.data[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.data[0] * v2.data[0], v1.data[1] * v2.data[1],
              v1.data[2] * v2.data[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.data[0] / v2.data[0], v1.data[1] / v2.data[1],
              v1.data[2] / v2.data[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t * v.data[0], t * v.data[1], t * v.data[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
  return vec3(v.data[0] / t, v.data[1] / t, v.data[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
  return vec3(t * v.data[0], t * v.data[1], t * v.data[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
  return v1.data[0] * v2.data[0] + v1.data[1] * v2.data[1] +
         v1.data[2] * v2.data[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
  return vec3((v1.data[1] * v2.data[2] - v1.data[2] * v2.data[1]),
              (-(v1.data[0] * v2.data[2] - v1.data[2] * v2.data[0])),
              (v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0]));
}

__host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v) {
  data[0] += v.data[0];
  data[1] += v.data[1];
  data[2] += v.data[2];
  return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &v) {
  data[0] *= v.data[0];
  data[1] *= v.data[1];
  data[2] *= v.data[2];
  return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const vec3 &v) {
  data[0] /= v.data[0];
  data[1] /= v.data[1];
  data[2] /= v.data[2];
  return *this;
}

__host__ __device__ inline vec3 &vec3::operator-=(const vec3 &v) {
  data[0] -= v.data[0];
  data[1] -= v.data[1];
  data[2] -= v.data[2];
  return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const float t) {
  data[0] *= t;
  data[1] *= t;
  data[2] *= t;
  return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const float t) {
  float k = 1.0 / t;

  data[0] *= k;
  data[1] *= k;
  data[2] *= k;
  return *this;
}

__host__ __device__ inline vec3 normalize(vec3 v) { return v / v.length(); }

#endif

#ifndef VEC2H
#define VEC2H

#include <iostream>
#include <math.h>

class vec2 {
public:
  __host__ __device__ vec2() {}
  __host__ __device__ vec2(float x, float y) {
    data[0] = x;
    data[1] = y;
  }
  __host__ __device__ inline float x() const { return data[0]; }
  __host__ __device__ inline float y() const { return data[1]; }
  __host__ __device__ inline float u() const { return data[0]; }
  __host__ __device__ inline float v() const { return data[1]; }

  __host__ __device__ inline const vec2 &operator+() const { return *this; }
  __host__ __device__ inline vec2 operator-() const {
    return vec2(-data[0], -data[1]);
  }
  __host__ __device__ inline float operator[](int i) const { return data[i]; }
  __host__ __device__ inline float &operator[](int i) { return data[i]; };

  __host__ __device__ inline vec2 &operator+=(const vec2 &v2);
  __host__ __device__ inline vec2 &operator-=(const vec2 &v2);
  __host__ __device__ inline vec2 &operator*=(const vec2 &v2);
  __host__ __device__ inline vec2 &operator/=(const vec2 &v2);
  __host__ __device__ inline vec2 &operator*=(const float t);
  __host__ __device__ inline vec2 &operator/=(const float t);

  __host__ __device__ inline float length() const {
    return sqrt(data[0] * data[0] + data[1] * data[1]);
  }
  __host__ __device__ inline float squared_length() const {
    return data[0] * data[0] + data[1] * data[1];
  }
  __host__ __device__ inline void make_unit_vector();

  float data[2];
};

inline std::istream &operator>>(std::istream &is, vec2 &t) {
  is >> t.data[0] >> t.data[1] >> t.data[2];
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec2 &t) {
  os << t.data[0] << " " << t.data[1] << " " << t.data[2];
  return os;
}

__host__ __device__ inline void vec2::make_unit_vector() {
  float k =
      1.0 / sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  data[0] *= k;
  data[1] *= k;
}

__host__ __device__ inline vec2 operator+(const vec2 &v1, const vec2 &v2) {
  return vec2(v1.data[0] + v2.data[0], v1.data[1] + v2.data[1]);
}

__host__ __device__ inline vec2 operator-(const vec2 &v1, const vec2 &v2) {
  return vec2(v1.data[0] - v2.data[0], v1.data[1] - v2.data[1]);
}

__host__ __device__ inline vec2 operator*(const vec2 &v1, const vec2 &v2) {
  return vec2(v1.data[0] * v2.data[0], v1.data[1] * v2.data[1]);
}

__host__ __device__ inline vec2 operator/(const vec2 &v1, const vec2 &v2) {
  return vec2(v1.data[0] / v2.data[0], v1.data[1] / v2.data[1]);
}

__host__ __device__ inline vec2 operator*(float t, const vec2 &v) {
  return vec2(t * v.data[0], t * v.data[1]);
}

__host__ __device__ inline vec2 operator/(vec2 v, float t) {
  return vec2(v.data[0] / t, v.data[1] / t);
}

__host__ __device__ inline vec2 operator*(const vec2 &v, float t) {
  return vec2(t * v.data[0], t * v.data[1]);
}

__host__ __device__ inline float dot(const vec2 &v1, const vec2 &v2) {
  return v1.data[0] * v2.data[0] + v1.data[1] * v2.data[1];
}

__host__ __device__ inline vec2 &vec2::operator+=(const vec2 &v) {
  data[0] += v.data[0];
  data[1] += v.data[1];
  return *this;
}

__host__ __device__ inline vec2 &vec2::operator*=(const vec2 &v) {
  data[0] *= v.data[0];
  data[1] *= v.data[1];
  return *this;
}

__host__ __device__ inline vec2 &vec2::operator/=(const vec2 &v) {
  data[0] /= v.data[0];
  data[1] /= v.data[1];
  return *this;
}

__host__ __device__ inline vec2 &vec2::operator-=(const vec2 &v) {
  data[0] -= v.data[0];
  data[1] -= v.data[1];
  return *this;
}

__host__ __device__ inline vec2 &vec2::operator*=(const float t) {
  data[0] *= t;
  data[1] *= t;
  return *this;
}

__host__ __device__ inline vec2 &vec2::operator/=(const float t) {
  float k = 1.0 / t;

  data[0] *= k;
  data[1] *= k;
  return *this;
}

__host__ __device__ inline vec2 normalize(vec2 v) { return v / v.length(); }

#endif

#ifndef VEC4H
#define VEC4H

#include <iostream>
#include <math.h>

class vec4 {

public:
  __host__ __device__ vec4() {}
  __host__ __device__ vec4(float x, float y, float z, float w) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
    data[3] = w;
  }
  __host__ __device__ inline float x() const { return data[0]; }
  __host__ __device__ inline float y() const { return data[1]; }
  __host__ __device__ inline float z() const { return data[2]; }
  __host__ __device__ inline float w() const { return data[3]; }
  __host__ __device__ inline float r() const { return data[0]; }
  __host__ __device__ inline float g() const { return data[1]; }
  __host__ __device__ inline float b() const { return data[2]; }
  __host__ __device__ inline float a() const { return data[3]; }

  __host__ __device__ inline const vec4 &operator+() const { return *this; }
  __host__ __device__ inline vec4 operator-() const {
    return vec4(-data[0], -data[1], -data[2], -data[3]);
  }
  __host__ __device__ inline float operator[](int i) const { return data[i]; }
  __host__ __device__ inline float &operator[](int i) { return data[i]; };

  __host__ __device__ inline vec4 &operator+=(const vec4 &v2);
  __host__ __device__ inline vec4 &operator-=(const vec4 &v2);
  __host__ __device__ inline vec4 &operator*=(const vec4 &v2);
  __host__ __device__ inline vec4 &operator/=(const vec4 &v2);
  __host__ __device__ inline vec4 &operator*=(const float t);
  __host__ __device__ inline vec4 &operator/=(const float t);

  __host__ __device__ inline float length() const {
    return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2] +
                data[3] * data[3]);
  }
  __host__ __device__ inline float squared_length() const {
    return data[0] * data[0] + data[1] * data[1] + data[2] * data[2] +
           data[3] * data[3];
  }
  __host__ __device__ inline void make_unit_vector();

  float data[4];
};

inline std::istream &operator>>(std::istream &is, vec4 &t) {
  is >> t.data[0] >> t.data[1] >> t.data[2] >> t.data[3];
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec4 &t) {
  os << t.data[0] << " " << t.data[1] << " " << t.data[2] << " " << t.data[3];
  return os;
}

__host__ __device__ inline void vec4::make_unit_vector() {
  float k = 1.0 / sqrt(data[0] * data[0] + data[1] * data[1] +
                       data[2] * data[2] + data[3] * data[3]);
  data[0] *= k;
  data[1] *= k;
  data[2] *= k;
  data[3] *= k;
}

__host__ __device__ inline vec4 operator+(const vec4 &v1, const vec4 &v2) {
  return vec4(v1.data[0] + v2.data[0], v1.data[1] + v2.data[1],
              v1.data[2] + v2.data[2], v1.data[3] + v2.data[3]);
}

__host__ __device__ inline vec4 operator-(const vec4 &v1, const vec4 &v2) {
  return vec4(v1.data[0] - v2.data[0], v1.data[1] - v2.data[1],
              v1.data[2] - v2.data[2], v1.data[3] - v2.data[3]);
}

__host__ __device__ inline vec4 operator*(const vec4 &v1, const vec4 &v2) {
  return vec4(v1.data[0] * v2.data[0], v1.data[1] * v2.data[1],
              v1.data[2] * v2.data[2], v1.data[3] * v2.data[3]);
}

__host__ __device__ inline vec4 operator/(const vec4 &v1, const vec4 &v2) {
  return vec4(v1.data[0] / v2.data[0], v1.data[1] / v2.data[1],
              v1.data[2] / v2.data[2], v1.data[3] / v2.data[3]);
}

__host__ __device__ inline vec4 operator*(float t, const vec4 &v) {
  return vec4(t * v.data[0], t * v.data[1], t * v.data[2], t * v.data[3]);
}

__host__ __device__ inline vec4 operator/(vec4 v, float t) {
  return vec4(v.data[0] / t, v.data[1] / t, v.data[2] / t, v.data[3] / t);
}

__host__ __device__ inline vec4 operator*(const vec4 &v, float t) {
  return vec4(t * v.data[0], t * v.data[1], t * v.data[2], t * v.data[3]);
}

__host__ __device__ inline float dot(const vec4 &v1, const vec4 &v2) {
  return v1.data[0] * v2.data[0] + v1.data[1] * v2.data[1] +
         v1.data[2] * v2.data[2] + v1.data[3] * v2.data[3];
}

__host__ __device__ inline vec4 &vec4::operator+=(const vec4 &v) {
  data[0] += v.data[0];
  data[1] += v.data[1];
  data[2] += v.data[2];
  data[3] += v.data[3];
  return *this;
}

__host__ __device__ inline vec4 &vec4::operator*=(const vec4 &v) {
  data[0] *= v.data[0];
  data[1] *= v.data[1];
  data[2] *= v.data[2];
  data[3] *= v.data[3];
  return *this;
}

__host__ __device__ inline vec4 &vec4::operator/=(const vec4 &v) {
  data[0] /= v.data[0];
  data[1] /= v.data[1];
  data[2] /= v.data[2];
  data[3] /= v.data[3];
  return *this;
}

__host__ __device__ inline vec4 &vec4::operator-=(const vec4 &v) {
  data[0] -= v.data[0];
  data[1] -= v.data[1];
  data[2] -= v.data[2];
  data[3] -= v.data[3];
  return *this;
}

__host__ __device__ inline vec4 &vec4::operator*=(const float t) {
  data[0] *= t;
  data[1] *= t;
  data[2] *= t;
  data[3] *= t;
  return *this;
}

__host__ __device__ inline vec4 &vec4::operator/=(const float t) {
  float k = 1.0 / t;

  data[0] *= k;
  data[1] *= k;
  data[2] *= k;
  data[3] *= k;

  return *this;
}

__host__ __device__ inline vec4 normalize(vec4 v) { return v / v.length(); }

#endif