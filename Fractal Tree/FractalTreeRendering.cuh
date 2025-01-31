// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <device_launch_parameters.h>

#include "KernelTree.cuh";
//#include "CPUTree.cuh";

void adjustCoordinateSystemOnResize(int width, int height);
void draw_func();
void renderTreeFromBuffer();