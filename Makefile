# Makefile for amalia_hier_leaf
#
# Usage:
#   make            build amalia_hier_leaf
#   make test       build, then run the core self-tests
#   make clean      remove build artifacts

CXX      := g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++11 -Isrc
LDFLAGS  := -fopenmp
LDLIBS   := -lm

# Optional NUMA-aware build, for a multi-socket/multi-node server (e.g.
# the 64-thread/128GB machine this was prepared for) - NOT needed on a
# single-node machine like this session's dev box (confirmed via
# `numactl --hardware`: 1 node here), where it would be a harmless but
# pointless no-op even if enabled. Requires libnuma installed
# (Debian/Ubuntu: apt install libnuma-dev). Usage: `make NUMA=1`.
ifdef NUMA
CXXFLAGS += -DUSE_NUMA
LDLIBS   += -lnuma
endif

# Optional GPU-accelerated BSGS hierarchy (Step 1: CPU<->GPU Bloom1
# bridge smoke test - --gpu-smoke-test). Requires the CUDA toolkit
# (nvcc) - NOT needed for the normal CPU-only build, which is
# unaffected either way. Usage: `make GPU=1`.
ifdef GPU
NVCC     := nvcc
NVCCFLAGS:= -O3
CXXFLAGS += -DGPU_ENABLED
GPU_OBJS := gpu_hierarchy.o
LINKER   := $(NVCC)
LDFLAGS  := -Xcompiler -fopenmp
else
LINKER   := $(CXX)
endif

TARGET   := amalia_hier_leaf

CXX_SRCS := amalia_hier_leaf.cpp \
            secp256k1/Int.cpp \
            secp256k1/IntMod.cpp \
            secp256k1/SECP256K1.cpp \
            secp256k1/Point.cpp \
            secp256k1/Random.cpp \
            hash/sha256.cpp \
            hash/ripemd160.cpp \
            hash/sha256_sse.cpp \
            hash/ripemd160_sse.cpp

C_SRCS   := util.c

CXX_OBJS := $(CXX_SRCS:.cpp=.o)
C_OBJS   := $(C_SRCS:.c=.o)
OBJS     := $(CXX_OBJS) $(C_OBJS) $(GPU_OBJS)

.PHONY: all clean test gen-test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LINKER) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# util.c actually uses C++ headers (<cstring>, etc.) despite the .c
# extension - must be compiled with g++, not gcc, or it fails to build.
%.o: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# GPU object - only relevant when built with `make GPU=1` (see GPU_OBJS
# above, empty otherwise, so this rule is simply never triggered in a
# normal CPU-only build).
gpu_hierarchy.o: gpu_hierarchy.cu gpu_hierarchy.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) gpu_hierarchy.o

# Core engine self-tests (normal + degenerate cases + upper-half case).
# Run after any change to the search loop before trusting a real attack.
test: $(TARGET)
	./$(TARGET) --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4
	./$(TARGET) --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4 --test-degenerate 1
	./$(TARGET) --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4 --test-degenerate 2
	./$(TARGET) --self-test --test-a-bits 24 --test-n 1024 --m-bits 20 -t 4 --test-degenerate 3

# Leaf-generation self-test (formula validity, sequential vs. direct-scalar
# path, and the random-batch equation check). Run before using --attack-loop.
gen-test: $(TARGET)
	./$(TARGET) --gen-self-test --tree-steps 19

# ============ amalia_worker: standalone worker binary ============
# A separate, trimmed-down source (amalia_worker.cpp) containing ONLY
# what --worker mode needs - no self-tests, no --master, no --build, no
# portfolio/radix4 experiments. Statically linked so it runs on any
# Linux machine without needing matching glibc/libstdc++/libgomp
# versions installed - the whole point of distributing this to workers
# across the internet rather than requiring them to build from source.
#
# KNOWN CAVEAT of full static linking: glibc's static NSS modules do not
# reliably resolve hostnames via getaddrinfo() in a statically-linked
# binary. --master-host should be given as a raw IP address, not a
# hostname, when using this binary - a hostname MAY still work on some
# systems but is not guaranteed across all of them.
WORKER_TARGET := amalia_worker
WORKER_CXX_SRCS := amalia_worker.cpp \
            secp256k1/Int.cpp \
            secp256k1/IntMod.cpp \
            secp256k1/SECP256K1.cpp \
            secp256k1/Point.cpp \
            secp256k1/Random.cpp \
            hash/sha256.cpp \
            hash/ripemd160.cpp \
            hash/sha256_sse.cpp \
            hash/ripemd160_sse.cpp
WORKER_C_SRCS := util.c
WORKER_CXX_OBJS := $(WORKER_CXX_SRCS:.cpp=.wo)
WORKER_C_OBJS := $(WORKER_C_SRCS:.c=.wo)
WORKER_OBJS := $(WORKER_CXX_OBJS) $(WORKER_C_OBJS)
WORKER_LDFLAGS := -static -fopenmp

.PHONY: worker worker-clean
worker: $(WORKER_TARGET)

$(WORKER_TARGET): $(WORKER_OBJS)
	$(CXX) $(WORKER_LDFLAGS) -o $@ $(WORKER_OBJS) $(LDLIBS)
	@echo "Built $(WORKER_TARGET) - verify with: file $(WORKER_TARGET) && ldd $(WORKER_TARGET)"
	@echo "(ldd should say 'not a dynamic executable' for a genuinely static binary)"

%.wo: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.wo: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

worker-clean:
	rm -f $(WORKER_OBJS) $(WORKER_TARGET)
