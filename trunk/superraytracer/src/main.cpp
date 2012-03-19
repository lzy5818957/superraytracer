
/*
 * Copyright:
 * Daniel D. Neilson (ddneilson@ieee.org)
 * University of Saskatchewan
 * All rights reserved
 *
 * Permission granted to use for use in assignments and
 * projects for CMPT 485 & CMPT 829 at the University
 * of Saskatchewan.
 */

#include "GL3/gl3w.h"
#include <GL/wglext.h>
#include <GL/glfw.h>

#include <cstdio>
#include <cstdlib>

#include "UI/ui.h"
#include "glUtils.h"
#include "demo.h"
#include "assign1.h"
#include "assign2.h"
#include "assign3.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);


int main(int argc, char *argv[])
{
	if ( !UI::init(640,480) || isGLError() )
	{
		fprintf(stderr, "ERROR: Could not initialize UI.\n");
		UI::shutdown();
		return EXIT_FAILURE;
	}

	// Print out some info on the runtime state
	fprintf(stdout, "GL VERSION: %s\n", glGetString(GL_VERSION));
	fprintf(stdout, "GLSL VERSION: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	fprintf(stdout, "Window settings:\n\trgba bits: (%d,%d,%d,%d)\n"
			"\tdepth bits: %d\n"
			"\tstencil bits: %d\n",
			glfwGetWindowParam(GLFW_RED_BITS),
			glfwGetWindowParam(GLFW_GREEN_BITS),
			glfwGetWindowParam(GLFW_BLUE_BITS),
			glfwGetWindowParam(GLFW_ALPHA_BITS),
			glfwGetWindowParam(GLFW_DEPTH_BITS),
			glfwGetWindowParam(GLFW_STENCIL_BITS));

	// Create & initialize the program object
	int choice=3;
#if 1
	do
	{
		printf("\n"
				"0) Demo\n"
				"1) Assignment 1\n"
				"2) Assignment 2\n"
				"3) Assignment 3\n"
				"Which program do you want to run? ");
		scanf("%d", &choice);
	} while (choice < 0 || choice > 3);
#endif
	UI::Callbacks *program;

	switch (choice)
	{
	case 0:
		program = new DemoProgram();
		if ( ! ((DemoProgram*)program)->init() )
		{
			fprintf(stderr, "Failed to initialize program\n");
			UI::shutdown();
			return EXIT_FAILURE;
		}
		// Set the window's title
		UI::setWindowTitle("Demo Program");
		break;

#define CASE_ASSIGNMENT_OBJ(N) \
	case N:\
	program = new Assignment##N(); \
	if ( ! ((Assignment##N*)program)->init() ) \
	{ \
		fprintf(stderr, "Failed to initialize program\n"); \
		UI::shutdown(); \
		return EXIT_FAILURE; \
	}\
	UI::setWindowTitle("Assignment " # N); \
	break;

	CASE_ASSIGNMENT_OBJ(1);
	CASE_ASSIGNMENT_OBJ(2);
	CASE_ASSIGNMENT_OBJ(3);
	}
	UI::setCallbacks(program);

	// Event loop
	UI::mainLoop();

	// Clean up the UI before exit
	UI::shutdown();

	delete program;

	return EXIT_SUCCESS;
}

/*
int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
*/