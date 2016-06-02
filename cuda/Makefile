#CUDADIR=/usr/local/cuda-5.0
#CUDADIR=/opt/cuda
#CUDA_SDK_DIR=/opt/cuda/sdk

INCFLAG=-I. -I/home/wittich/src/tbb43_20150424oss/include -I$(CUDADIR)/include # -I$(CUDA_SDK_DIR)/C/common/inc -I..


# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM352   := -gencode=arch=compute_35,code=compute_35
#GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
#GENCODE_FLAGS   :=  $(GENCODE_SM13) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM352)
GENCODE_FLAGS   :=  $(GENCODE_SM35)

NVCC = nvcc
NVCCFLAGS = -O3 $(INCFLAG)   $(GENCODE_FLAGS) --ptxas-options=-v -lineinfo  -std=c++11 
#NVCCFLAGS = -G -g $(INCFLAG)   $(GENCODE_FLAGS) --ptxas-options=-v -lineinfo  -std=c++11

NVCCFLAGS += -Xcompiler="-ftree-vectorizer-verbose=4 -ftree-vectorize "#-fopt-info-vec-missed"

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

test: test.o
	$(NVCC) $^ -o $@ 
m2: m2.o
	$(CXX) $(CXXFLAGS) $^ -o $@ 


ROOTF="-I$(shell root-config --incdir)"
NVCCFLAGS += $(ROOTF)
stest: stest.o
	$(NVCC) $(NVCCFLAGS)  -ltbb $^ -o $@ 

clean:
	$(RM) *.o *~ 

depend:
	makedepend  *.cu *.cc *.h

# DO NOT DELETE

mtest.o: Time.hh Matriplex.h
nbody.o: timer.h cuda_math.h
nbody_nothrust.o: /usr/include/sys/stat.h /usr/include/features.h
nbody_nothrust.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
nbody_nothrust.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
nbody_nothrust.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
nbody_nothrust.o: /usr/include/time.h /usr/include/bits/stat.h
nbody_nothrust.o: /usr/include/sys/types.h /usr/include/endian.h
nbody_nothrust.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
nbody_nothrust.o: /usr/include/sys/select.h /usr/include/bits/select.h
nbody_nothrust.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
nbody_nothrust.o: /usr/include/sys/sysmacros.h
nbody_nothrust.o: /usr/include/bits/pthreadtypes.h /usr/include/stdio.h
nbody_nothrust.o: /usr/include/libio.h /usr/include/_G_config.h
nbody_nothrust.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
nbody_nothrust.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
nbody_nothrust.o: /usr/include/bits/waitflags.h
nbody_nothrust.o: /usr/include/bits/waitstatus.h /usr/include/alloca.h
nbody_nothrust.o: /usr/include/math.h /usr/include/bits/huge_val.h
nbody_nothrust.o: /usr/include/bits/huge_valf.h /usr/include/bits/huge_vall.h
nbody_nothrust.o: /usr/include/bits/inf.h /usr/include/bits/nan.h
nbody_nothrust.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
nbody_nothrust.o: /usr/include/sys/time.h /usr/include/signal.h
nbody_nothrust.o: /usr/include/bits/signum.h /usr/include/bits/siginfo.h
nbody_nothrust.o: /usr/include/bits/sigaction.h
nbody_nothrust.o: /usr/include/bits/sigcontext.h /usr/include/bits/sigstack.h
nbody_nothrust.o: /usr/include/sys/ucontext.h /usr/include/bits/sigthread.h
