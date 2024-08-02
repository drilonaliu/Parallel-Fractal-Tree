#include "FractalTree.cuh";

GLuint bufferObj;
cudaGraphicsResource* resource;
int iteration = 0;
int numVertices;
int iterations;

void setUpCudaOpenGLInterop();
void bindFunctionsToWindow();

void startFractalTree(int argc, char** argv) {
	printControls();
	initializeWindow(argc, argv);
	setUpCudaOpenGLInterop();
	bindFunctionsToWindow();
	glutMainLoop();
}

void initializeWindow(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(512, 512);
	glutCreateWindow("Fractal Tree");
	createMenu();
	glewInit();
}

void bindFunctionsToWindow() {
	glutSpecialFunc(specialKeyHandler);
	glutKeyboardFunc(keyboardHandler);
	glutReshapeFunc(adjustCoordinateSystemOnResize);
	glutDisplayFunc(draw_func);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
	glutMouseWheelFunc(mouseWheel);
}

void setUpCudaOpenGLInterop() {
	//Choose the most suitable CUDA device based on the specified properties(in prop) for openGL.
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaError_t error = cudaChooseDevice(&dev, &prop);
	if (error != cudaSuccess) {
		printf("Error choosing CUDA device: %s\n", cudaGetErrorString(error));
	}
	cudaGLSetGLDevice(dev);

	//Buffer Size
	iterations = 10;
	numVertices = numberOfVertices(25);
	size_t bufferSize = 2 * numVertices * sizeof(float); //each point has 2 components x,y 

	//Generate openGL buffer
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObj); //Set the context of this buffer obj. In our case its a vertex obj buffer
	glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_DYNAMIC_COPY); 

	//Notify CUDA runtime that we intend to share the OpenGL buffer named bufferObj with CUDA.//FlagsNone, ReadOnly, WriteOnly
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}


int numberOfVertices(int iteration) {
	return 2 * (pow(2, iteration + 1) - 1);//each segment has 2 points 
}