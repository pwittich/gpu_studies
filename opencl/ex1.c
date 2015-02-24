#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
 
////////////////////////////////////////////////////////////////////////////////
 
// Use a static data size for simplicity
//
#define DATA_SIZE (1024)
 
////////////////////////////////////////////////////////////////////////////////
 
// Simple compute kernel that computes the square of an input array.       [1]
//
const char *KernelSource = "\n" \
  "__kernel square(                                                       \n" \
  "   __global float* input,                                              \n" \
  "   __global float* output,                                             \n" \
  "   const unsigned int count)                                           \n" \
  "{                                                                      \n" \
  "   int i = get_global_id(0);                                           \n" \
  "   if(i < count)                                                       \n" \
  "       output[i] = input[i] * input[i];                                \n" \
  "}                                                                      \n" \
  "\n";
 
////////////////////////////////////////////////////////////////////////////////
 
int main(int argc, char** argv)
{
  int err;                          // error code returned from api calls
 
  float data[DATA_SIZE];            // original data set given to device
  float results[DATA_SIZE];         // results returned from device
  unsigned int correct;             // number of correct results returned
 
  size_t global;                    // global domain size for our calculation
  size_t local;                     // local domain size for our calculation
 
  cl_device_id device_id;           // device ID
  cl_context context;               // context
  cl_command_queue queue;           // command queue
  cl_program program;               // program
  cl_kernel kernel;                 // kernel
 
  cl_mem input;                     // device memory used for the input array
  cl_mem output;                    // device memory used for the output array
 
  // Get data on which to operate
  //
 
  int i = 0;
  unsigned int count = DATA_SIZE;
  for(i = 0; i < count; i++)
    data[i] = ...;
  ...
 
    // Get an ID for the device                                    [2]
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1,
			 &device_id, NULL);
    if (err != CL_SUCCESS)
      { ... }                                                        //      [3]
 
    // Create a context                                            [4]
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
      { ... }
 
    // Create a command queue                                              [5]
    //
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue)
      { ... }
 
    // Create the compute program from the source buffer                   [6]
    //
    program = clCreateProgramWithSource(context, 1,
					(const char **) & KernelSource, NULL, &err);
    if ( program)
      { ... }
 
    // Build the program executable                                        [7]
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
      {
        size_t len;
        char buffer[2048];
 
        printf("Error: Failed to build program executable\n");             [8]
									     clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
												   sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
      }
 
    // Create the compute kernel in the program we wish to run            [9]
    //
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
      { ... }
 
    // Create the input and output arrays in device memory for our calculation
    //                                                                   [10]
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) *count,
			   NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *count,
			    NULL, NULL);
    if (!input || !output)
      { ... }
 
    // Write our data set into the input array in device memory          [11]
    //
    err = clEnqueueWriteBuffer(queue, input, CL_TRUE, 0,
			       sizeof(float) *count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
      { ... }
 
    // Set the arguments to our compute kernel                           [12]
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
      { ... }
 
    // Get the maximum work-group size for executing the kernel on the device
    //                                                                   [13]
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
				   sizeof(int), &local, NULL);
    if (err != CL_SUCCESS)
      { ... }
 
    // Execute the kernel over the entire range of the data set          [14]
    //
    global = count;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
				 0, NULL, NULL);
    if (err)
      { ... }
 
    // Wait for the command queue to get serviced before reading back results
    //                                                                   [15]
    clFinish(queue);
 
    // Read the results from the device                                  [16]
    //
    err = clEnqueueReadBuffer(queue, output, CL_TRUE, 0,
			      sizeof(float) *count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
      { ... }
 
    // Shut down and clean up
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    return 0;
}
 
