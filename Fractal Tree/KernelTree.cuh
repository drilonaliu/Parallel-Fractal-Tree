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

#include <cooperative_groups.h>;
__device__ struct Point {
	float x;
	float y;
};

__device__  struct Branch {
	Point start;
	Point end;
	float length;
};

__device__  struct Triangle {
	Point A;
	Point B;
	Point C;
};

__global__ void divideBranch(float* points, Branch branch, float angle_left, float angle_right, int iteration, int max_iterations, int id);
__global__ void branchDivide(float* points, Branch branch, Branch* branches, float angle_left, float angle_right, int start_iteration, int max_iterations, int threadShiftIndex);
__device__ Branch makeChildBranch(Branch parentBranch, float angle);