
#CC=clang
CFLAGS=-O3
CXXFLAGS=-O3
LIBSDIR=

all: nbody


nbody: nbody.o
	$(CXX) $(CXXFLAGS) -ltbb -o $@ $<
