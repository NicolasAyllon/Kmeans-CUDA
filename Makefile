CC = g++
CXXOPTS = -std=c++17 -Wall -Werror -lpthread -O2
NVCCOPTS = -arch=sm_75

SDIR = src
NVSDIR = src_cuda
INC = -I src
CUDA_INC = -I src_cuda
CUDA_INC += -I NVIDIA_SDK_Samples/common/inc
CUDA_INC += -I /usr/local/cuda/include
ODIR = obj
XDIR = bin

# Make directories for object files
$(shell mkdir -p obj/cpu)
$(shell mkdir -p obj/cuda)
$(shell mkdir -p obj/cuda_shared)

# Object file lists
# Separate into lists (by folder) for cpu, cuda, cuda_shared, and thrust
OBJS_CPU = $(addprefix obj/cpu/, argparse.o io.o kmeans.o main.o random.o)
OBJS_CUDA = $(addprefix obj/cuda/, argparse.o io.o kmeans_kernels.o main.o random.o)
OBJS_CUDA_SHARED = $(addprefix obj/cuda_shared/, argparse.o io.o kmeans_kernels.o main.o random.o)

all: sequential

sequential: $(OBJS_CPU)
	mkdir -p bin
	$(CC) $(ODIR)/cpu/*.o $(CXXOPTS) $(INC) -o $(XDIR)/kmeans_cpu

cuda: $(OBJS_CUDA)
	mkdir -p bin
	nvcc $(ODIR)/cuda/*.o $(NVCCOPTS) $(INC) $(CUDA_INC) -o $(XDIR)/kmeans_cuda 

cuda_shared: $(OBJS_CUDA_SHARED)
	mkdir -p bin
	nvcc $(ODIR)/cuda_shared/*.o $(NVCCOPTS) $(INC) $(CUDA_INC) -o $(XDIR)/kmeans_cuda_shared

thrust:

clean:
	rm -f $(XDIR)/*

# Create object file (.o) from source (.cpp)
# % matches a pattern (argparse, io, kmeans, main, ...)
# $< expands to 1st item in prerequisite list
# $@ expands to target

# Different versions of main by preprocessor define (-D) flags
$(ODIR)/cpu/main.o: $(SDIR)/main.cpp
	$(CC) $(INC) $(CUDA_INC) -D CPU -c $< -o $@

$(ODIR)/cuda/main.o: $(SDIR)/main.cpp
	nvcc $(NVCCOPTS) $(INC) $(CUDA_INC) -D CUDA -c $< -o $@

$(ODIR)/cuda_shared/main.o: $(SDIR)/main.cpp
	nvcc $(NVCCOPTS) $(INC) $(CUDA_INC) -D CUDA_SHARED -c $< -o $@

$(ODIR)/thrust/main.o: $(SDIR)/main.cpp
	nvcc $(NVCCOPTS) $(INC) $(CUDA_INC) -D THRUST -c $< -o $@


# Kernels
$(ODIR)/cuda/kmeans_kernels.o: $(NVSDIR)/kmeans_kernels.cu
	nvcc $(NVCCOPTS) $(INC) $(CUDA_INC) -D CUDA -c $< -o $@

$(ODIR)/cuda_shared/kmeans_kernels.o: $(NVSDIR)/kmeans_kernels.cu
	nvcc $(NVCCOPTS) $(INC) $(CUDA_INC) -D CUDA_SHARED -c $< -o $@


# Other object files
$(ODIR)/cpu/%.o: $(SDIR)/%.cpp
	$(CC) $(INC) $(CUDA_INC) -c $< -o $@

$(ODIR)/cuda/%.o: $(SDIR)/%.cpp
	$(CC) $(INC) $(CUDA_INC) -c $< -o $@

$(ODIR)/cuda_shared/%.o: $(SDIR)/%.cpp
	$(CC) $(INC) $(CUDA_INC) -c $< -o $@

$(ODIR)/thrust/%.o: $(SDIR)/%.cpp
	$(CC) $(INC) $(CUDA_INC) -c $< -o $@
