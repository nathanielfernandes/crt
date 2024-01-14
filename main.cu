#include "include/camera.cu"
#include "include/lodepng.h"
#include "include/vec3.cu"

#include <cstdlib>
#include <iostream>
#include <time.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ vec3 color(const ray &r) {
  vec3 unit_direction = normalize(r.direction());
  float t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y, camera **cam) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_x) || (j >= max_y))
    return;

  int pixel_index = j * max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);

  ray r = (*cam)->get_ray(u, 1.0 - v);

  fb[pixel_index] = color(r);
}

void save_fb(const std::string &filename, const vec3 *fb, int w, int h,
             int num_channels = 3) {
  std::vector<unsigned char> image(4 * w * h);
  for (size_t i = 0; i < w * h; ++i) {
    image[4 * i + 0] = 255 * fb[i].r();
    image[4 * i + 1] = 255 * fb[i].g();
    image[4 * i + 2] = 255 * fb[i].b();
    image[4 * i + 3] = 255;
  }
  unsigned error = lodepng::encode(filename, image, w, h);
  if (error)
    std::cout << "PNG encoder error " << error << ": "
              << lodepng_error_text(error) << std::endl;

  std::cout << "Saved " << filename << std::endl;
}

__global__ void create_camera(camera **d_cam, int nx, int ny) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_cam = new camera(vec3(0, 0, 1), vec3(0, 0, -1), vec3(0, 1, 0), 20.0,
                        float(nx) / float(ny));
  }
}

int main() {

  const int nx = 1600;
  const int ny = 900;
  const int tx = 2;
  const int ty = 2;

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  const int num_pixels = nx * ny;
  const size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // create camera
  camera **d_cam;
  checkCudaErrors(cudaMallocManaged((void **)&d_cam, sizeof(camera *)));
  create_camera<<<1, 1>>>(d_cam, nx, ny);

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  render<<<blocks, threads>>>(fb, nx, ny, d_cam);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::string filename("out.png");
  save_fb(filename, fb, nx, ny);

  // clean up
  checkCudaErrors(cudaFree(fb));
  checkCudaErrors(cudaFree(d_cam));
}
