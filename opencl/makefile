CC=gcc
CFLAGS=-O3 -I ${CUDADIR}/include  -std=gnu99
CXXFLAGS=-O3 -I ${CUDADIR}/include 
OPENCLLIBS=-L${CUDADIR}/lib64 -lOpenCL
LIBS=$(OPENCLLIBS) -lm

all: nbody_simple nbody nbody2


nbody_simple: nbody_simple.o
	$(CXX) $(CXXFLAGS) -o $@ $<

nbody: nbody.o
	$(CC) $(CFLAGS) $(LIBS) -o $@ $<

nbody2: nbody2.o
	$(CC) $(CFLAGS) $(LIBS) -o $@ $<

dumpcl: dumpcl.o
	$(CC) $(CFLAGS) $(LIBS) -o $@ $<

printcl: printcl.o
	$(CC) $(CFLAGS) $(LIBS) -o $@ $<


clean:
	$(RM) *.o *~
