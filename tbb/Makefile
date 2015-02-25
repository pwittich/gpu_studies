
#CC=clang
CFLAGS=-O3
CXXFLAGS=-O3 -std=c++11
LIBSDIR=

all: nbody


nbody: nbody.o
	$(CXX) $(CXXFLAGS) -ltbb -o $@ $<
