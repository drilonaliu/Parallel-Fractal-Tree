// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
//Fractal Tree
#include "FractalTreeVariables.cuh";
#include "FractalTreeRendering.cuh";
#include "UserInteraction.cuh";

using namespace std;

void startFractalTree(int argc, char** argv);
int numberOfVertices(int iteration);
void initializeWindow(int argc, char** argv);
