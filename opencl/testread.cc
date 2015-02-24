#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#include <math.h>

struct rgb_t {
  //unsigned char r,g,b, padding; // 8 bit depth, padded to 32 bits
  unsigned char b, g, r;
} ;


int main()
{

  
  FILE* fin = fopen("otest2.bmp", "r");
  if ( fin == (FILE*)NULL ) {
    fprintf(stderr, "writeBMP: warning, couldn't open output file.\n");
    return 1;
  }
  unsigned char header[14];
  fread(header, sizeof(unsigned char), 14, fin);
  unsigned char dib_header[40];
  fread(dib_header, sizeof(unsigned char), 40, fin);

  unsigned int *sz; sz = (unsigned int*)(header+2);
  size_t size = sz[0];
  printf("size = %lu\n", size);	
  int img_size  = size - 54;

  unsigned int *dibh = (unsigned int*)(dib_header);

  int width  = dibh[1];
  int height = dibh[2];
  printf("width, heigh = %d x %d\n", width, height);

  char *img = (char*)malloc(img_size*sizeof(char));
  
  fread(img, sizeof(char), img_size, fin);

  struct rgb_t *vals = (struct rgb_t*)img;
  int pos = height/2;
  //int pos = 0;

  for (int i = 0; i < height; ++i ) {
    printf("r %i: ", i);
    for (int j = 0; j < width; ++j ) {
      int ind = i*width + j;
      printf("%d: (%d,%d,%d) ", j, vals[ind].r, vals[ind].g, vals[ind].b);
    }
    printf("\n");
  }
  free(img);
  return 0;

}
