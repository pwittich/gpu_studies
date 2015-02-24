#include <iostream>
#include <cstdio>

#include "bitmap.hh"

using namespace std;

int main()
{

  CBitmap a;
  bool pass = a.Load("otest2.bmp");
  if ( ! pass ) {
    cerr << "Load failed: " << endl;
    return 1;
  }

  RGBA *data = a.GetBits();
  printf ("data = %p\n", data);
  int width = a.GetWidth();
  int height = a.GetHeight();
  cout << "height, width = " << height <<  ", " << width << endl;

  for (int i = 0; i < height; ++i ) {
    for ( int j =0; j < width; ++j ) {
      int ind = i*width+j;
      printf("(%d, %d) = (%d,%d,%d) ", i,j,data[ind].Red, data[ind].Green, data[ind].Blue);//, data[ind].Alpha)
    }
    
  }
  return 0;
}
