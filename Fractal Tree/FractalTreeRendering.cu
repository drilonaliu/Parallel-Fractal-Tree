#include "FractalTreeRendering.cuh";
#include "FractalTreeVariables.cuh";
#include "CPUTree.cuh";


#include <chrono>
using namespace std;
using namespace std::chrono;


float angle_left = 3.14f / 12.0f;
float angle_right = -3.14f / 12.0f;
bool animation = !true;
bool color = false;
bool rightAngle = true;
bool leftAngle = true;

bool CPUImplementation = false;
bool GPUImplementationNormal = false;
bool GPUImplementationFast = true;
bool GPUImplementationRecursive = false;

bool generatePoints = true;

bool runOnlyOnce = true;
Branch* d_branches;
Branch b;
float* devPtr;

void generatePointsUsing100GPU();
void generatePointsUsingNormalGPU();;
void generatePointsUsingRecursiveGPU();
void clearBackground();
void setUpInitialTreeBranch();
int numberOfVertices2(int iteration);

/*
* Main drawing function binded to the GLUT window.
*/
void draw_func() {
	clearBackground();
	setUpInitialTreeBranch();

	if (GPUImplementationFast) {
		if (generatePoints) {
			generatePointsUsing100GPU();
		}
		renderTreeFromBuffer();
	}
	else if (CPUImplementation) {
		drawFractalTreeCPU();
	}
	else if (GPUImplementationNormal) {
		if (generatePoints) {
			generatePointsUsingNormalGPU();
		}
		renderTreeFromBuffer();
	}
	else if (GPUImplementationRecursive) {
		generatePointsUsingRecursiveGPU();
		renderTreeFromBuffer();
	}
}

void clearBackground() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
}

void adjustCoordinateSystemOnResize(int width, int height) {
	glViewport(0, 0, width, height);
	float aspect = (1.0f * width) / (1.0f * height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-1.6 * aspect, 1.6 * aspect, -1.7, 1.5);
	glutPostRedisplay();
}

void setUpInitialTreeBranch() {
	b.start.x = 0.0f;
	b.start.y = -1.15;
	b.end.x = 0.0f;
	b.end.y = -0.60f;
	b.length = 0.55f;
}

/*
* Makes a call to kernel to generate fractal tree points.
* Each kernel call will have 1 block and 1024 threads.
*/
void generatePointsUsingNormalGPU() {
	int threads = 1024;
	int blocks = 1;
	int maxNumberOfThreads = blocks * threads;;
	
	//Allocate memory for graph tree branch in GPU
	if (runOnlyOnce) {
		cudaMalloc((void**)&d_branches, (pow(2, 26) - 1) * sizeof(Branch));
		runOnlyOnce = false;
	}


	size_t size;

	//Map resource to OpenGL
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	//Launching Kernels
	int numTotalThreads = round(pow(2, iterations + 1)) - 1;
	int kernelCalls = numTotalThreads / (blocks * threads);
	int threadShiftIndex = 0;
	int start_iteration = 1;

	//Keep calling the kernel until we generate all the data
	for (int k = 0; k <= kernelCalls; k++) {
		threadShiftIndex = k * maxNumberOfThreads;
		start_iteration = (log2(threadShiftIndex));
		if (threadShiftIndex == 0) {
			start_iteration = 1;
		}
		branchDivide << <blocks, threads >> > (devPtr, b, d_branches, angle_left, angle_right, start_iteration, iterations, threadShiftIndex);
	}

	//End of Kernel Calls
	cudaGraphicsUnmapResources(1, &resource, NULL);
}

/*
* Makes a call to kernel to generate fractal tree points.
* Each kernel call will use every thread in GPU, hence using 
* 100% of GPU.
*/
void generatePointsUsing100GPU() {
	//Find the number of threads and blocks needed for cooperative groups
	int threads;
	int blocks;
	int maxNumberOfThreads;
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, branchDivide, 0, 0);
	maxNumberOfThreads = blocks * threads;

	//Allocate memory for graph tree branch in GPU
	if (runOnlyOnce) {
		cudaMalloc((void**)&d_branches, (pow(2, 26) - 1) * sizeof(Branch));
		runOnlyOnce = false;
	}

	size_t size;

	//Map resource to OpenGL
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	//Launching Kernels
	int numTotalThreads = round(pow(2, iterations + 1)) - 1;
	int kernelCalls = numTotalThreads / (blocks * threads);
	int threadShiftIndex = 0;
	int start_iteration = 1;

//	auto start = high_resolution_clock::now();
	//Keep calling the kernel until we generate all the data
	for (int k = 0; k <= kernelCalls; k++) {
		threadShiftIndex = k * maxNumberOfThreads;
		start_iteration = (log2(threadShiftIndex));
		if (threadShiftIndex == 0) {
			start_iteration = 1;
		}
		void* kernelArgs[] = { &devPtr, &b, &d_branches, &angle_left, &angle_right, &start_iteration, &iterations,&threadShiftIndex };
		cudaLaunchCooperativeKernel((void*)branchDivide, blocks, threads, kernelArgs, 0, 0);
	}

	//auto stop = high_resolution_clock::now();
	//auto duration = duration_cast<microseconds>(stop - start);

	//cout << "\nTime taken by function: "
	//	<< duration.count() << " microseconds" << endl;

	//End of kernel calls
	cudaGraphicsUnmapResources(1, &resource, NULL);
}

/*
* Makes a call to recursive generating points kernel in GPU.
* 
*/
void generatePointsUsingRecursiveGPU() {
	//cudaGraphicsMapResources(1, &resource, NULL);
	//cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);
	//divideBranch << <1, 2 >> > (devPtr, b, angle_left, angle_right, 0, iterations, 1);
	//cudaDeviceSynchronize();
	//cudaGraphicsUnmapResources(1, &resource, NULL);
}

void renderTreeFromBuffer() {
	//Register has float data, point has 2 components
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(0);

	//Changing line width and color for iterations.
	glColor3f(1.0f, 1.0f, 1.0f);
	int a = numberOfVertices2(6);

	if (iterations > 6) {
		glLineWidth(4);
		if (color) {
			glColor3f(0.45f, 0.23f, 0.10f);
		}
		glDrawArrays(GL_LINES, 0, a);

		glLineWidth(3);
		if (color) {
			glColor3f(0.21f, 0.44f, 0.10f);
		}
		glDrawArrays(GL_LINES, a, (numberOfVertices2(iterations)) - a);
	}
	else {
		if (color) {
			glColor3f(0.45f, 0.23f, 0.10f);
		}
		glDrawArrays(GL_LINES, 0, numberOfVertices2(iterations));
	}
	glutSwapBuffers();

	if (animation) {
		if (rightAngle) {
			angle_right += 0.01f;
		}
		if (leftAngle) {
			angle_left -= 0.01f;
		}
		glutPostRedisplay();
	}
}

int numberOfVertices2(int iteration) {
	return 2 * (pow(2, iteration + 1) - 1);
}
