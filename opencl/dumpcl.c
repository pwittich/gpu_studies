// clang -framework OpenCL dumpcl.c -o dumpcl && ./dumpcl

#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

int main(int argc, char* const argv[]) {
    cl_uint num_devices, i;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    cl_device_id* devices = calloc(sizeof(cl_device_id), num_devices);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    const size_t buf_size = 10240;
    char buf[buf_size];
    cl_uint val;
    cl_ulong uval;
    for (i = 0; i < num_devices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, buf_size, buf, NULL);
        fprintf(stdout, "Device %s supports ", buf);

        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, buf_size, buf, NULL);
        fprintf(stdout, "%s,", buf);

        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &val, NULL);
        fprintf(stdout, "%u comput units,", val);

        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &val, NULL);
        fprintf(stdout, "%u max cloq freq,", val);

        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &uval, NULL);
        fprintf(stdout, "%llu Mb global memory size,", uval/1024/1024);

        clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &uval, NULL);
        fprintf(stdout, "%llu kb local memory size", uval/1024);

	fprintf(stdout, "\n");


    }

    // 
    cl_platform_id platforms[10];
    cl_uint num_platforms;
    clGetPlatformIDs(10, platforms, &num_platforms);
    fprintf(stdout, "num_platforms = %u\n" , num_platforms);
    for ( i = 0; i < num_platforms; ++i ) {
      cl_int err = clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, buf_size, buf, NULL);
      fprintf(stdout, "Platform Profile =\t %s\n", buf);
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, buf_size, buf, NULL);
      fprintf(stdout, "Platform Version =\t %s\n", buf);

      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, buf_size, buf, NULL);
      fprintf(stdout, "Platform Name =   \t %s\n", buf);

      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, buf_size, buf, NULL);
      fprintf(stdout, "Platform Vendor = \t %s\n", buf);

      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, buf_size, buf, NULL);
      fprintf(stdout, "Platform Extensions =\t %s\n", buf);

    }

    free(devices);
}
