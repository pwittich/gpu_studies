// based on transpose.c by apple
// clang -framework OpenCL nbody.c

#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#include <math.h>

const char* oclErrorString(cl_int error);

void writeDat(const char* fname, void* data, int ndata);
//void writeBMP(const char* fname, const char *data);

/////////////////////////////////////////////////////////////////////////////

static char    *
load_program_source(const char *filename)
{
  struct stat     statbuf;
  FILE           *fh;
  char           *source;

  fh = fopen(filename, "r");
  if (fh == 0)
    return 0;

  stat(filename, &statbuf);
  source = (char *) malloc(statbuf.st_size + 1);
  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0';

  return source;
}

/////////////////////////////////////////////////////////////////////////////
const int width  = 512;
const int height = 512;

int 
main(int argc, char **argv)
{
  uint64_t        t0, t1, t2;
  int             err;
  cl_device_id    device_id[2];
  cl_context      context;
  cl_kernel       kernel;
  cl_command_queue queue;
  cl_program      program;
  cl_mem          bpos, bvel; //, bpblock;


  // steering inputs

  int step,burst;

  int nparticle = 512; /* MUST be a nice power of two for simplicity */
  //int nparticle = 1024; // nice power of 2 for simplicity
  const size_t nstep = 1250;
  int nburst = 500; // sub-steps
  int nthread = 64; 

  float dt = 0.0005;
  float eps = 0.001;

  // create arrays for local variables
  cl_float4 *pos = (cl_float4*)malloc(sizeof(cl_float4)*nparticle);
  cl_float4 *vel  = (cl_float4*)malloc(sizeof(cl_float4)*nparticle);

  srand(42137377L);

  // two black holes
  float offset = +175.;
  pos[0].x = 0.5*width ;
  pos[0].y = 0.5*height;
  pos[0].z = 0.;
  pos[0].w = 0;//5000.; // mass
  pos[1].x = 3.*width ;
  pos[1].y = 0.5*height;
  pos[1].z = 0.;
  pos[1].w = 0;//40000.; // mass
  pos[2].x = 3.*width ;
  pos[2].y = 0.75*height;
  pos[2].z = 0.;
  pos[2].w = 0.;//40000.; // mass
  pos[3].x = 3.*width ;
  pos[3].y = 0.75*height;
  pos[3].z = 0.;
  pos[3].w = 0.;//40000.; // mass
  

  for ( int i = 4; i < nparticle; ++i ){
    bool done = false;
    float mult = 1.0;
    // if ( i > (nparticle/2) ) 
    //   mult = -1.0;
    //while (! done ) {
      pos[i].x = 1.0*rand()/RAND_MAX * width;// - width/2. ;
      pos[i].y = 1.0*rand()/RAND_MAX * height;// - height/2. ;
      pos[i].z = 1.0*rand()/RAND_MAX * 50. -  25.;
      pos[i].w = 1;//1.0*rand()/RAND_MAX * 5.; // mass
      //pos[i].w = 2;
      //float rsq = pos[i].x*pos[i].x+pos[i].y*pos[i].y+pos[i].z*pos[i].z;
      //if ( (rsq < (40*40)) && (rsq > (35*35))) 
    //   if ((rsq < (40*40)) || 1) {
    // 	done = true;
    // 	pos[i].x += mult*offset;
    // 	pos[i].y += mult*offset;
    //   }
    // }
    //pos2[i] = (cl_float4){0.f,0.f,0.f,0.f};
    vel[i]  = (cl_float4){0.f,0.f,0.f,0.f};
  }
  writeDat("start.bmp", (void*)pos, nparticle);

  int which = 8;
  printf("Start: particle %d x=%f, y=%f, z=%f, m=%f\n",
	 which, pos[which].x, pos[which].y, pos[which].z, pos[which].w);
  cl_float4 startpos = pos[which];

  // for ( int j =120; j < 130; ++j ) {
  //   printf("Start1:   particle %d x=%f, y=%f, z=%f, m=%f\n",
  // 	   j, pos[j].x, pos[j].y, pos[j].z, pos[j].w);
  // }
  // for ( int j =120; j < 130; ++j ) {
  //   printf("Start2:   particle %d x=%f, y=%f, z=%f, m=%f\n",
  // 	   j, pos2[j].x, pos2[j].y, pos2[j].z, pos2[j].w);
  // }
  // for ( int j =120; j < 130; ++j ) {
  //   printf("Start3:   particle %d x=%f, y=%f, z=%f, m=%f\n",
  // 	   j, vel[j].x, vel[j].y, vel[j].z, vel[j].w);
  // }

  //Connect to a GPU compute device
  //
  int gpu = 1;
  unsigned int num_dev_returned;
  err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 2, device_id, &num_dev_returned);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to create a device group! (%s, line %d)\n", oclErrorString(err), __LINE__);
    return EXIT_FAILURE;
  }

  // what am I running on 
  char param[100];
  size_t ret_size;
  int device_id_offset = 0;
  clGetDeviceInfo(device_id[device_id_offset],CL_DEVICE_NAME,sizeof(param),param,&ret_size);
  printf("Running on Device %s\n",param);


  //Create a compute context
  //
  context = clCreateContext(0, 1, device_id+device_id_offset, NULL, NULL, &err);
  if (!context) {
    printf("Error: Failed to create a compute context! (%s)\n", oclErrorString(err));
    return EXIT_FAILURE;
  }

  //Create a command queue
  //
  queue = clCreateCommandQueue(context, device_id[device_id_offset], 0, &err);
  if (!queue) {
    printf("Error: Failed to create a command queue! (%s)\n", oclErrorString(err));
    return EXIT_FAILURE;
  }

  //Load the compute program from disk into a cstring buffer
  //
  char   *source = load_program_source("nbody_k2.cl");
  if (!source) {
    printf("Error: Failed to load compute program from file!\n");
    return EXIT_FAILURE;
  }
  //printf("Program follows:\n%s\n", source);

  //Create the compute program from the source buffer
  //
  program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &err);
  if (!program || err != CL_SUCCESS) {
    printf("Error: Failed to create compute program!(%s)\n", oclErrorString(err));
    return EXIT_FAILURE;
  }
  free(source); // yes virginia this is required

  //Build the program executable
  //
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    // JIT 'compile' error messages will appear here
    printf("Error: Failed to build program executable!\n");
    size_t          len;
    char            buffer[2048];
    clGetProgramBuildInfo(program, device_id[device_id_offset], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("Program Build Info:\n--\n%s\n--\n", buffer);

    return EXIT_FAILURE;
  }

  // Create the compute kernel from within the program
  //
  kernel = clCreateKernel(program, "nbody_k2", &err);
  if (!kernel || err != CL_SUCCESS) {
    printf("Error: Failed to create compute kernel!(%s)\n", oclErrorString(err));
    return EXIT_FAILURE;
  }

  // create arrays for pos, pos2, velocity
  bpos = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * nparticle, NULL, NULL);
  bvel  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * nparticle, NULL, NULL);

  if ( ! ( bpos && bvel ) ) {
    printf("Error: Failed to allocate source arrays!\n");
    return EXIT_FAILURE;
  }

  //Fill the input array with the host allocated random data
  //
  err = clEnqueueWriteBuffer(queue, bpos, true, 0, sizeof(cl_float4) * nparticle, 
			     pos, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array! (%s)\n", oclErrorString(err));
    return EXIT_FAILURE;
  }
  err = clEnqueueWriteBuffer(queue, bvel, true, 0, sizeof(cl_float4) * nparticle, 
			     vel, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array! (%s)\n", oclErrorString(err));
    return EXIT_FAILURE;
  }

  //Create the output array on the device
  //
  //dst = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * (height + PADDING), NULL, NULL);
  //  if (!dst) {
  // printf("Error: Failed to allocate destination array!\n");
  // return EXIT_FAILURE;
  // }

  //Determine the global and local dimensions for the execution
  //
  size_t global, local;
  global          = nparticle;
  local = 256;

  //Set the kernel arguments prior to execution
  //
  err  = clSetKernelArg(kernel, 0, sizeof(float), &dt);
  err |= clSetKernelArg(kernel, 1, sizeof(float), &eps);
  err |= clSetKernelArg(kernel, 2, sizeof(float), &nburst);

  err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bpos);
  err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bvel);
  err |= clSetKernelArg(kernel, 5, sizeof(cl_float4)*global, NULL); // local buffer.
  err |= clSetKernelArg(kernel, 6, sizeof(cl_float4)*global, NULL); // local buffer.

  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments! (%s)\n", oclErrorString(err));
    return EXIT_FAILURE;
  }


  // Start the timing loop and execute the kernel over several iterations
  //
  printf("starting ... \n");

  double tavg_0 = 0, tsqu_0 = 0;

  for ( int iter = 0; iter < nstep;++iter) {
    printf("iter=%d\n", iter);
    int             k;
    err = CL_SUCCESS;
    t0 = t1 = mach_absolute_time();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

    if ( err != CL_SUCCESS ) {
      printf("Error: Failed to execute kernel! k=%d, (%s)\n", k, oclErrorString(err));    
      break;
    }
    clFinish(queue);
    t2 = mach_absolute_time();
    if (err != CL_SUCCESS) {
      printf("Error: Failed to execute kernel!\n");
      return EXIT_FAILURE;
    }
    //printf("done.\n");
    //Calculate the total bandwidth that was obtained on the device for all  memory transfers
    //
    struct mach_timebase_info info;
    mach_timebase_info(&info);
    double          t = 1e-9 * (t2 - t1) * info.numer / info.denom;
    //printf("Time spent = %g\n", t);

    tavg_0 += t;
    tsqu_0 += t*t;

    //Read back the results that were computed on the device
    //
    err = clEnqueueReadBuffer(queue, bpos, true, 0, sizeof(cl_float4)*nparticle , pos, 
			      0, NULL, NULL);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to read back results from the device! (%s)\n", 
	     oclErrorString(err));
      return EXIT_FAILURE;
    }



    if ( iter%25==0 ) {    
      // get velocities
      err = clEnqueueReadBuffer(queue, bvel, true, 0, sizeof(cl_float4)*nparticle , vel, 0, NULL, NULL);
      if (err != CL_SUCCESS) {
	printf("Error: Failed to read back results from the device! (%s)\n", oclErrorString(err));
	return EXIT_FAILURE;
      }

      // calculate total momentum
      cl_float4 ptotv = (cl_float4){0.,0.,0.,0.};
      for ( int i = 0; i < nparticle; ++i ) {
	ptotv.x += pos[i].w*vel[i].x;
	ptotv.y += pos[i].w*vel[i].y;
	ptotv.z += pos[i].w*vel[i].z;
      }
      float ptot = sqrt(ptotv.x*ptotv.x+ptotv.y*ptotv.y+ptotv.z*ptotv.z);
      printf("\t\tptot = %5.3f\n", ptot);
    }

     printf("End:   particle %d x=%f, y=%f, z=%f, m=%f\n",
     	 which, pos[which].x, pos[which].y, pos[which].z, pos[which].w);
    // printf("End:   particle %d x=%f, y=%f, z=%f, m=%f\n",
    // 	 which, pos2[which].x, pos2[which].y, pos2[which].z, pos2[which].w);

    // for ( int j =120; j < 130; ++j ) {
    //   printf("End2:   particle %d x=%f, y=%f, z=%f, m=%f\n",
    // 	   j, pos[j].x, pos[j].y, pos[j].z, pos[j].w);
    // }

    // for ( int j =120; j < 130; ++j ) {
    //   printf("End2:   particle %d x=%f, y=%f, z=%f, m=%f\n",
    // 	   j, pos2[j].x, pos2[j].y, pos2[j].z, pos2[j].w);
    // }

    // for ( int j =120; j < 130; ++j ) {
    //   printf("End3:   particle %d x=%f, y=%f, z=%f, m=%f\n",
    // 	   j, vel[j].x, vel[j].y, vel[j].z, vel[j].w);
    // }

    cl_float4 endpos = pos[which];
  
    cl_float4 sep = (cl_float4){endpos.x-startpos.x,endpos.y-startpos.y,endpos.z-startpos.z,0};
    float distance = sqrt(sep.x*sep.x + sep.y*sep.y + sep.z*sep.z);
    //printf("Distance travelled = %g\n", distance);

    char fname[256];
    sprintf(fname, "/var/tmp/wittich/test_%d.bmp", iter);
    writeDat(fname, (void*)pos, nparticle);
  }

  double tavg = tavg_0/nstep/nburst;
  double trms = sqrt((tsqu_0- tavg_0*tavg_0/(nstep*nburst))/(nstep*nburst-1));;
  printf("Average time spent per single step  = %g pm %g\n", tavg, trms);


  free(pos);
  free(vel);
  //free(h_data);
  //free(h_result);

  clReleaseMemObject(bpos);
  clReleaseMemObject(bvel);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);


  return 0;
}



////////////////////////////////////////
// Helper function to get error string
// *********************************************************************
const char* oclErrorString(cl_int error)
{
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "";

}

struct rgb_t {
  //unsigned char r,g,b, padding; // 8 bit depth, padded to 32 bits
  unsigned char b, g, r;
} ;

void writeDat(const char* fname, void* data, int ndata)
{
  //fprintf(stdout,"writing to %s\n", fname);
  FILE* fout = fopen(fname, "w");
  if ( fout == (FILE*)NULL ) {
    fprintf(stderr, "writeBMP: warning, couldn't open output file.\n");
    return ;
  }


  const int header_size = 14;
  unsigned char header[header_size];
  bzero(header, header_size);
  header[0] = 'B';
  header[1] = 'M';
  uint *ha = (uint*)(header+2); // 2 byte offset
  ha[2] = 54; // offset; 40+14; constant (two headers)

  const int dib_header_size = 40;
  unsigned char dib_header[dib_header_size];
  bzero(dib_header, dib_header_size);

  uint *a = (uint*)(dib_header);
  unsigned short int *b = (unsigned short int*)(dib_header);
  a[0] = dib_header_size;
  a[1] = width; // width
  a[2] = height; // height
  b[6] = 1; // must be 1
  b[7] = 24; // color depth
  a[4] = 0; // compression method - 0 is default 
  a[5] = 0; // image size - can be zero
  a[6] = 100; // vertical resolution, pixels/m, signed (?)
  a[7] = 100; // horizontal resolution, pixels/m, signed (?)
  a[8] = 0; // colormap entries
  a[9] = 0; // important colors. ignored.
  int padding = (sizeof(struct rgb_t)*width)%4;

  struct rgb_t vals[height][width]; // I hate c
  bzero(vals, width*height*sizeof(struct rgb_t));
  //printf("size = %lu\n", width*height*sizeof(struct rgb_t)+header_size+dib_header_size);
  //printf("data size = %lu\n", width*height*sizeof(struct rgb_t));
  //printf("rgb_t size = %lu\n", sizeof(struct rgb_t));
  ha[0] = width*height*sizeof(struct rgb_t)+header_size+dib_header_size; // size of the file
  //printf("size=%d\n", ha[0]);
  // for ( int i = 0; i < header_size; ++i ) {
  //   printf("c[%d] = %d\n", i, header[i]);
  // }


  for ( int i = 0; i < width; ++i ) {
    for ( int j = 0; j < height; ++j ) {
      vals[j][i].r = 0;
      vals[j][i].g = 0;//4*j;
      vals[j][i].b = 0;
    }
  }

  // int pos = height/2;
  // printf("pos = %d\n", pos);
  // for ( int i = 0; i < width; ++i ) {
  //   for (int j = -10; j < 10; ++j ) {
  //     vals[pos+j][i].r = 255;
  //     vals[pos+j][i].g = 255;
  //     vals[pos+j][i].b = 255;
  //   }
  //   //printf("%d,%d,%d,%d\n", i, vals[i][pos].r, vals[i][pos].g, vals[i][pos].b);
  // }

  


  cl_float4 *cdata = (cl_float4*)data;
  int cnt = 0;
  for ( int i = 0; i < ndata; ++i ) {
    int x = (cdata[i].x+0.5); //x += width/2; //x = x%width;
    int y = (cdata[i].y+0.5); //y += height/2; //y = y%height;
    //printf("i=%d,x,y = %d, %d\n", i,x, y);
    if ( x < 0 || y < 0 ) continue;
    if ( (x >= width) || (y >= height) ) continue;
    ++cnt;
    vals[y][x].r = 255;
    vals[y][x].g = 255;
    vals[y][x].b = 255;
  }
  printf("File written (%d particles).\n", cnt);

  // for ( int i = 0; i < ndata; ++i ) 
  //   fprintf(fout, "%f %f %f\n", cdata[i].x,cdata[i].y,cdata[i].z);
  
  fwrite(header, sizeof(char), header_size, fout);
  fwrite(dib_header, sizeof(char), dib_header_size, fout);
  fwrite(vals, sizeof(struct rgb_t), width*height, fout);
  //fwrite(cdata, sizeof(cl_float4), ndata, fout);

  fclose(fout);
  return;

}

