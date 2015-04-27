#CUDADIR=/usr/local/cuda-5.0
#CUDADIR=/opt/cuda
#CUDA_SDK_DIR=/opt/cuda/sdk

INCFLAG=-I$(CUDADIR)/include # -I$(CUDA_SDK_DIR)/C/common/inc -I..


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
#NVCCFLAGS = -O3 $(INCFLAG) --ptxas-options=-v -use_fast_math $(GENCODE_FLAGS)
#NVCCFLAGS = -O3 $(INCFLAG)  -use_fast_math $(GENCODE_FLAGS)
NVCCFLAGS = -O3 $(INCFLAG)   $(GENCODE_FLAGS) --ptxas-options=-v 
#NVCCFLAGS = -g $(INCFLAG) --verbose $(GENCODE_SM20) #-DTHRUST_DEBUG

#NVCCFLAGS = -O3 $(INCFLAG)  --use_fast_math $(GENCODE_SM13) #use to compile on l2gpu with GTX285
#NVCCFLAGS = -O3 $(INCFLAG)  --use_fast_math $(GENCODE_SM20)
#NVCCFLAGS = -g -lineinfo -DTHRUST_DEBUG   $(INCFLAG)   $(GENCODE_FLAGS) #-Xcompiler "-std=c++11"


#LIBS=-L/home/wittich/src/cudpp_install_2.0/lib -lcudpp

%.o: %.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: nbody test nbody_nothrust


nbody: nbody.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ 
nbody_nothrust: nbody_nothrust.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ 

test: test.o
	$(NVCC) $^ -o $@ 


clean:
	$(RM) *.o *~ 

depend:
	makedepend  *.cu *.cc *.h

# DO NOT DELETE

nbody.o: timer.h cuda_math.h
