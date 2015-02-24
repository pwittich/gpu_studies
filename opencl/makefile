CC=clang
CFLAGS=-O3
CXXFLAGS=-O3

all: nbody_simple nbody nbody2


nbody_simple: nbody_simple.o
	$(CXX) $(CXXFLAGS) -o $@ $<

nbody: nbody.o
	$(CC) $(CFLAGS) -framework OpenCL -o $@ $<

nbody2: nbody2.o
	$(CC) $(CFLAGS) -framework OpenCL -o $@ $<
dumpcl: dumpcl.o
	$(CC) $(CFLAGS) -framework OpenCL -o $@ $<
printcl: printcl.o
	$(CC) $(CFLAGS) -framework OpenCL -o $@ $<

clean:
	$(RM) *.o *~
