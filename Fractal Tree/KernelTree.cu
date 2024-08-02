#include "KernelTree.cuh";
namespace cg = cooperative_groups;

/*
* Kernel Method for generating the points for the fractal tree in an iterative way.
* In each iteration, 2^iteration threads will work in parallel.
* Uses cooperative groups for synchronization of all threads in the GPU.
* 
* @points - device pointer for the buffer points that will be rendered from OpenGL.
* @branch - initial branch of the tree.
* @branches - array that will be used to represent the tree graph.
* @angle_left - the angle by which each child branch deviates from the parent branch in the left direction.
* @angle_right - the angle by which each child branch deviates from the parent branch in the right direction.
* @start_iteration - starting iteration that threads should start generetaing branches.
* @max_iterations - the iteration to which the tree will be grown.
* @threadShiftIndex - used for mapping threads to branches.
*/
__global__ void branchDivide(float* points, Branch branch, Branch* branches, float angle_left,
	float angle_right, int start_iteration, int max_iterations, int threadShiftIndex) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;;
	idx += threadShiftIndex;

	Branch childBranch;
	Branch parentBranch;
	float angle;
	
	auto g = cg::this_grid();

	if (idx == 0) {
		points[0] = branch.start.x;
		points[1] = branch.start.y;
		points[2] = branch.end.x;
		points[3] = branch.end.y;
		branches[1] = branch;
	}

	for (int iteration = start_iteration; iteration <= max_iterations; iteration++) {
		float start_at = round(pow(2, iteration));
		int end_at = round((pow(2, iteration + 1))) - 1;

		if (idx >= start_at && idx <= end_at) {
			int parentNode = idx / 2;
			parentBranch = branches[parentNode];

			int t = idx % 2;
			if (t == 0) {
				angle = angle_left;
			}
			else {
				angle = angle_right;
			}

			childBranch = makeChildBranch(parentBranch,angle);
			branches[idx] = childBranch;

			//add points to points array;
			int offset = 2 * 2 * (idx - 1);
			points[offset] = childBranch.start.x;
			points[offset + 1] = childBranch.start.y;
			points[offset + 2] = childBranch.end.x;
			points[offset + 3] = childBranch.end.y;
		}
		g.sync();
	//	__syncthreads();
	}
}

/*
* Kernel Method for generating the points for the fractal tree in a recursive way.
* Each block works with 2 threads that will divide the branch in parallel.
* Each block recursivly launches another kernel for the child branches.
* 
* @points - device pointer for the buffer points that will be rendered from OpenGL.
* @branch - initial branch of the tree.
* @angle_left - each left branch will be this degrees far from the parent branch.
* @angle_right - each rigt branch will be this degrees far from the parent branch.
* @iteration - starting iteration that threads should start generetaing branches.
* @max_iterations - the iteration to which the tree will be grown.
* @nodeId - used for mapping the threads to branches.
*/
__global__ void divideBranch(float* points, Branch branch, float angle_left, float angle_right, int iteration, int max_iterations, int nodeId) {
	int idx = threadIdx.x; // Since we are launching  2 threads, this is sufficient
	Branch smolBranch;
	Point start;
	Point end;
	Point A;
	float angle;

	//Iterimi i pare
	if (iteration == 0) {
		if (idx == 0) {
			points[0] = branch.start.x;
			points[1] = branch.start.y;
			points[2] = branch.end.x;
			points[3] = branch.end.y;

		}
	}
	__syncthreads();

	//Divide Branch
	if (idx == 0) {
		angle = angle_left;
	}
	else {
		angle = angle_right;
	}

	smolBranch = makeChildBranch(branch, angle);

	nodeId = 2 * nodeId + idx;
	int offset = 2 * 2 * (nodeId - 1);
	points[offset] = smolBranch.start.x;
	points[offset + 1] = smolBranch.start.y;
	points[offset + 2] = smolBranch.end.x;
	points[offset + 3] = smolBranch.end.y;

	iteration += 1;
	if (iteration < max_iterations) {
		divideBranch << <1, 2 >> > (points, smolBranch, angle_left, angle_right, iteration, max_iterations, nodeId);
	}
}

/*
* Makes a child branch of the parent brach for the fractal tree
* 
* @parentBranch - parent branch from which a child branch will be created
* @angle - the angle by which each child branch deviates from the parent branch in the left direction.
*/
__device__ Branch makeChildBranch(Branch parentBranch, float angle) {
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
