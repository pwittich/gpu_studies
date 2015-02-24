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

void writeDat(const char* fname, void* data, int ndata)
{
  FILE* fout = fopen(fname, "w");
  if ( fout == (FILE*)NULL ) {
    fprintf(stderr, "writeBMP: warning, couldn't open output file.\n");
    return ;
  }

  const int width = 100;
  const int height = 200;

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
  // for ( int ii = 0; ii < 10; ++ii ) 
  //   printf("a[%d] = 0x%x\n", ii,a[ii]);
  // for ( int ii = 0; ii < dib_header_size; ++ii ) 
  //   printf("ca[%d] = 0%o\n", ii,dib_header[ii]);
  int padding = (sizeof(rgb_t)*width)%4;
  printf("padding = %d\n", padding);

  struct rgb_t vals[height][width]; // I hate c
  bzero(vals, width*height*sizeof(struct rgb_t));
  //printf("size = %lu\n", width*height*sizeof(struct rgb_t)+header_size+dib_header_size);
  printf("data size = %lu\n", width*height*sizeof(struct rgb_t));
  printf("rgb_t size = %lu\n", sizeof(struct rgb_t));
  ha[0] = width*height*sizeof(struct rgb_t)+header_size+dib_header_size; // size of the file
  printf("size=%d\n", ha[0]);
  // for ( int i = 0; i < header_size; ++i ) {
  //   printf("c[%d] = %d\n", i, header[i]);
  // }


  for ( int i = 0; i < width; ++i ) {
    for ( int j = 0; j < height; ++j ) {
      vals[j][i].r = 255;
      vals[j][i].g = 0;//4*j;
      vals[j][i].b = 0;
    }
  }

  int pos = height/2;
  printf("pos = %d\n", pos);
  for ( int i = 0; i < width; ++i ) {
    for (int j = -10; j < 10; ++j ) {
      vals[pos+j][i].r = 255;
      vals[pos+j][i].g = 255;
      vals[pos+j][i].b = 255;
    }
    //printf("%d,%d,%d,%d\n", i, vals[i][pos].r, vals[i][pos].g, vals[i][pos].b);
  }

  


  // cl_float4 *cdata = (cl_float4*)data;
  // for ( int i = 0; i < ndata; ++i ) {
  //   int x = (cdata[i].x); x += 64; x = x%width;
  //   int y = (cdata[i].y); y += 64; y = y%height;
  //   printf("x,y = %d, %d\n", x, y);
  //   if ( x < 0 || y < 0 ) continue;
  //   vals[x][y].r = 0;
  //   vals[x][y].g = 0;
  //   vals[x][y].b = 0;
  // }

  //printf("rgb = %u%u%u\n", vals[10][10].r,vals[10][10].g,vals[10][10].b);
  

  // for ( int i = 0; i < ndata; ++i ) 
  //   fprintf(fout, "%f %f %f\n", cdata[i].x,cdata[i].y,cdata[i].z);
  
  fwrite(header, sizeof(char), header_size, fout);
  fwrite(dib_header, sizeof(char), dib_header_size, fout);
  fwrite(vals, sizeof(struct rgb_t), width*height, fout);
  //fwrite(cdata, sizeof(cl_float4), ndata, fout);

  fclose(fout);
  return;

}

int main()
{	
  int testdata[100];
  bzero(testdata, 100*4);
  writeDat("otest.bmp", testdata, 100);
  return 0;
}
