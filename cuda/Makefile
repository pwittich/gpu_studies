# to allow cross compiling on the ARM/Jetson TK1
ifdef TK1
# Jetson TK1
$(info compling for TK1/ARM)
ARCH=ARM
BITS=32
NVCC_C=arm-linux-gnueabihf-g++
else
# x86
$(info compiling for non-TK1)
ARCH=x86
BITS=64
NVCC_C=g++
endif

INCFLAG=-I. -I/home/wittich/src/tbb43_20150424oss/include -I$(CUDAINCDIR)


# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 
GENCODE_SM32    := -gencode arch=compute_32,code=sm_32  # Tegra K1
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM352   := -gencode=arch=compute_35,code=compute_35
#GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
#GENCODE_FLAGS   :=  $(GENCODE_SM13) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM352)
#GENCODE_FLAGS   :=  $(GENCODE_SM35) $(GENCODE_SM32)
GENCODE_FLAGS   :=  $(GENCODE_SM32)

NVCC = nvcc
NVCCFLAGS = -O3 $(INCFLAG)   $(GENCODE_FLAGS) --ptxas-options=-v -lineinfo  -std=c++11 
#NVCCFLAGS = -G -g $(INCFLAG)   $(GENCODE_FLAGS) --ptxas-options=-v -lineinfo  -std=c++11

NVCCFLAGS += -Xcompiler="-ftree-vectorizer-verbose=1 -ftree-vectorize "#-fopt-info-vec-missed"
NVCCFLAGS += -target-cpu-arch=${ARCH} --machine=${BITS} -ccbin arm-linux-gnueabihf-g++ 

CXXFLAGS =  $(INCFLAG) -std=c++11


#LIBS=-L/home/wittich/src/cudpp_install_2.0/lib -lcudpp

%.o: %.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: nbody nbody_nothrust mtest stest

mtest: mtest.o
	$(NVCC) $(NVCCFLAGS) -ltbb $^ -o $@ 

mtest2: mtest2.o
	$(CXX) $(CXXFLAGS) $^ -o $@ 


nbody: nbody.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ 
nbody_nothrust: nbody_nothrust.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ 

nbody_unified: nbody_unified.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ 

test: test.o
	$(NVCC) $^ -o $@ 
m2: m2.o
	$(CXX) $(CXXFLAGS) $^ -o $@ 


ROOTF="-I$(shell root-config --incdir)"
#NVCCFLAGS += $(ROOTF)
stest: stest.o
	$(NVCC) $(NVCCFLAGS)  -ltbb $^ -o $@ 

clean:
	$(RM) *.o *~ 

depend:
	makedepend  *.cu *.cc *.h

# DO NOT DELETE

