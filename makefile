ifeq ($(OS),Windows_NT)
	detected_OS := Windows
else
	detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif

LFLAGS += -lfftw3 -lm

ifeq ($(detected_OS),Darwin)
	LFLAGS += -framework Accelerate
else
	LFLAGS += -lcblas
endif

CXXFLAGS += -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -g 
CPPFLAGS += -std=c++14
OBJS =  obj/tensor_calc.o obj/tensor_conv.o obj/tensor_blas.o obj/autodiff.o obj/net.o obj/layers.o 

all: net
net: obj/main.o
	$(CXX) $^ -o $@ $(LFLAGS) 

PHONY: .test
test: obj/test.o $(OBJS)
	$(CXX) $^ -o $@ $(LFLAGS)

obj/%.o: src/%.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

PHONY: .clean
clean:
	rm obj/*

