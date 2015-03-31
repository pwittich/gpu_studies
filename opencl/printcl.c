//#include <OpenCL/opencl.h> // APPLE
#include <CL/opencl.h>

#include <errno.h>
#include <stdio.h>


/*Display OpenCL system info */
int main()
{
  const size_t MAX_DEVICES=10;
  size_t p_size;
  size_t arr_tsize[3];
  size_t ret_size;
  char param[512];
  cl_uint entries;
  cl_ulong long_entries;
  cl_bool bool_entries;
  cl_device_id devices[MAX_DEVICES];
  cl_uint num_devices;
  cl_device_local_mem_type mem_type;
  cl_device_type dev_type;
  cl_device_fp_config fp_conf;
  cl_device_exec_capabilities exec_cap;

   

  clGetPlatformInfo(NULL, CL_PLATFORM_PROFILE,sizeof(param),param,&ret_size);
  printf("\nPlatform Profile:\t%s\n",param);
  clGetPlatformInfo(NULL, CL_PLATFORM_VERSION,sizeof(param),param,&ret_size);
  printf("Platform Version:\t%s\n",param);

  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL,MAX_DEVICES,devices,&num_devices);
  printf("Found Devices:\t\t%d\n",num_devices);

  for(int i=0; i<num_devices; i++)
    {
      printf("\nDevice: %d\n\n",i);
   
      clGetDeviceInfo(devices[i],CL_DEVICE_TYPE,sizeof(dev_type),&dev_type,&ret_size);
      printf("\tDevice Type:\t\t");
      if(dev_type & CL_DEVICE_TYPE_GPU)
	printf("CL_DEVICE_TYPE_GPU ");
      if(dev_type & CL_DEVICE_TYPE_CPU)
	printf("CL_DEVICE_TYPE_CPU ");
      if(dev_type & CL_DEVICE_TYPE_ACCELERATOR)
	printf("CL_DEVICE_TYPE_ACCELERATOR ");
      if(dev_type & CL_DEVICE_TYPE_DEFAULT)
	printf("CL_DEVICE_TYPE_DEFAULT ");
      printf("\n");


      clGetDeviceInfo(devices[i],CL_DEVICE_NAME,sizeof(param),param,&ret_size);
      printf("\tName: \t\t\t%s\n",param);

      clGetDeviceInfo(devices[i],CL_DEVICE_VENDOR,sizeof(param),param,&ret_size);
      printf("\tVendor: \t\t%s\n",param);

      clGetDeviceInfo(devices[i],CL_DEVICE_VENDOR_ID,sizeof(cl_uint),&entries,&ret_size);
      printf("\tVendor ID:\t\t%d\n",entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_VERSION,sizeof(param),param,&ret_size);
      printf("\tVersion:\t\t%s\n",param);
      
      clGetDeviceInfo(devices[i],CL_DEVICE_PROFILE,sizeof(param),param,&ret_size);
      printf("\tProfile:\t\t%s\n",param);
   
      clGetDeviceInfo(devices[i],CL_DRIVER_VERSION,sizeof(param),param,&ret_size);
      printf("\tDriver: \t\t%s\n",param);

      clGetDeviceInfo(devices[i],CL_DEVICE_EXTENSIONS,sizeof(param),param,&ret_size);
      printf("\tExtensions:\t\t%s\n",param);

      clGetDeviceInfo(devices[i],CL_DEVICE_MAX_WORK_ITEM_SIZES,3*sizeof(size_t),arr_tsize,&ret_size);
      printf("\tMax Work-Item Sizes:\t(%lu,%lu,%lu)\n",arr_tsize[0],arr_tsize[1],arr_tsize[2]);

      clGetDeviceInfo(devices[i],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&p_size,&ret_size);
      printf("\tMax Work Group Size:\t%lu\n",p_size);
      
      clGetDeviceInfo(devices[i],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&entries,&ret_size);
      printf("\tMax Compute Units:\t%d\n",entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(cl_uint),&entries,&ret_size);
      printf("\tMax Frequency (MHz):\t%d\n",entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,sizeof(cl_uint),&entries,&ret_size);
      printf("\tCache Line (bytes):\t%d\n",entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&long_entries,&ret_size);
      printf("\tGlobal Memory (MB):\t%llu\n",long_entries/1024/1024);

      clGetDeviceInfo(devices[i],CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong),&long_entries,&ret_size);
      printf("\tLocal Memory (kB):\t%llu\n",long_entries/1024);

      clGetDeviceInfo(devices[i],CL_DEVICE_LOCAL_MEM_TYPE,sizeof(cl_device_local_mem_type),&mem_type,&ret_size);
      if(mem_type & CL_LOCAL)
	printf("\tLocal Memory Type:\tCL_LOCAL\n");
      else if(mem_type & CL_GLOBAL)
	printf("\tLocal Memory Type:\tCL_GLOBAL\n");
      else
	printf("\tLocal Memory Type:\tUNKNOWN\n");


      clGetDeviceInfo(devices[i],CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&long_entries,&ret_size);
      printf("\tMax Mem Alloc (MB):\t%llu\n",long_entries/1024/1024);

      clGetDeviceInfo(devices[i],CL_DEVICE_MAX_PARAMETER_SIZE,sizeof(size_t),&p_size,&ret_size);
      printf("\tMax Param Size (MB):\t%lu\n",p_size);

      clGetDeviceInfo(devices[i],CL_DEVICE_MEM_BASE_ADDR_ALIGN,sizeof(cl_uint),&entries,&ret_size);
      printf("\tBase Mem Align (bits):\t%d\n",entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_ADDRESS_BITS,sizeof(cl_uint),&entries,&ret_size);
      printf("\tAddress Space (bits):\t%d\n",entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_IMAGE_SUPPORT,sizeof(cl_bool),&bool_entries,&ret_size);
      printf("\tImage Support:\t\t%d\n",bool_entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_TYPE,sizeof(fp_conf),&fp_conf,&ret_size);
      printf("\tFloat Functionality:\t");
      if(fp_conf & CL_FP_DENORM)
	printf("DENORM support ");
      if(fp_conf & CL_FP_ROUND_TO_NEAREST)
	printf("Round to nearest support ");
      if(fp_conf & CL_FP_ROUND_TO_ZERO)
	printf("Round to zero support ");
      if(fp_conf & CL_FP_ROUND_TO_INF)
	printf("Round to +ve/-ve infinity support ");
      if(fp_conf & CL_FP_FMA)
	printf("IEEE754 fused-multiply-add support ");
      if(fp_conf & CL_FP_INF_NAN)
	printf("INF and NaN support ");
      printf("\n");


      clGetDeviceInfo(devices[i],CL_DEVICE_ERROR_CORRECTION_SUPPORT,sizeof(cl_bool),&bool_entries,&ret_size);
      printf("\tECC Support:\t\t%d\n",bool_entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_EXECUTION_CAPABILITIES,sizeof(cl_device_exec_capabilities),&exec_cap,&ret_size);
      printf("\tExec Functionality:\t");
      if(exec_cap & CL_EXEC_KERNEL)
	printf("CL_EXEC_KERNEL ");
      if(exec_cap & CL_EXEC_NATIVE_KERNEL)
	printf("CL_EXEC_NATIVE_KERNEL ");
      printf("\n");

      clGetDeviceInfo(devices[i],CL_DEVICE_ENDIAN_LITTLE,sizeof(cl_bool),&bool_entries,&ret_size);
      printf("\tLittle Endian Device:\t%d\n",bool_entries);

      clGetDeviceInfo(devices[i],CL_DEVICE_PROFILING_TIMER_RESOLUTION,sizeof(size_t),&p_size,&ret_size);
      printf("\tProfiling Res (ns):\t%lu\n",p_size);

      clGetDeviceInfo(devices[i],CL_DEVICE_AVAILABLE,sizeof(cl_bool),&bool_entries,&ret_size);
      printf("\tDevice Available:\t%d\n",bool_entries);

    }
}
