
#CC=clang
CFLAGS=-O3
CXXFLAGS=-O3 -std=c++11 -g
LIBSDIR=

all: nbody nbody_soa

nbody: nbody.o
	$(CXX) $(CXXFLAGS) -ltbb -o $@ $<

nbody_soa: nbody_soa.o
	$(CXX) $(CXXFLAGS) -ltbb -o $@ $<

clean:
	$(RM) *.o nbody nbody_soa
