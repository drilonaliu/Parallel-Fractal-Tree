#include "CPUTree.cuh"
#include "KernelTree.cuh";

#include <chrono>
using namespace std;
using namespace std::chrono;


const int arraySize = (pow(2, 27) - 1);
Branch* branches = new Branch[arraySize];
Branch makeChildBranchCPU(Branch parentBranch, float angle);


void drawFractalTreeCPU() {
	glClear(GL_COLOR_BUFFER_BIT);
	glPointSize(1.0f);
	glColor3f(0.0f, 0.31f, 0.45f); // coolblue

	//First Branch
	Branch b;
	b.start.x = 0.0f;
	b.start.y = -1.15;
	b.end.x = 0.0f;
	b.end.y = -0.60f;
	b.length = 0.55f;
	branches[1] = b;

	//Draw the first initial branch.
	glBegin(GL_LINES);
	glVertex2f(b.start.x, b.start.y);
	glVertex2f(b.end.x, b.end.y);
	glEnd();

	Branch childBranch;
	Branch parentBranch;
	float angle;

	// Divide and draw algorithm
	int numThreads = (pow(2, iterations) - 1);

	for (int idx = 2; idx <= numThreads; idx++) {
		int parentNode = idx / 2;
		parentBranch = branches[parentNode];
		int direction = idx % 2;

		if (direction == 0) {
			angle = angle_left;
		}
		else {
			angle = angle_right;
		}

		childBranch = makeChildBranchCPU(parentBranch, angle);
		branches[idx] = childBranch;

		//Draw the child branch branch
		glBegin(GL_LINES);
		glVertex2f(childBranch.start.x, childBranch.start.y);
		glVertex2f(childBranch.end.x, childBranch.end.y);
		glEnd();
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

Branch makeChildBranchCPU(Branch parentBranch, float angle) {
	Point start;
	Point end;
	Point A;
	Branch childBranch;
	start = parentBranch.end;
	float k = 1.25f;
	A.x = (parentBranch.end.x * (1 + k) - parentBranch.start.x) / k;
	A.y = (parentBranch.end.y * (1 + k) - parentBranch.start.y) / k;
	end.x = A.x * cos(angle) - A.y * sin(angle) - start.x * cos(angle) + start.y * sin(angle) + start.x;
	end.y = A.x * sin(angle) + A.y * cos(angle) - start.x * sin(angle) - start.y * cos(angle) + start.y;
	childBranch.start = start;
	childBranch.end = end;
	return childBranch;
}
