CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2

all: main_stationary main_nonstationary

main_stationary: main_stationary.cpp idat.hpp clustering_evaluation_metrics.hpp
	$(CXX) $(CXXFLAGS) -o $@ main_stationary.cpp

main_nonstationary: main_nonstationary.cpp idat.hpp clustering_evaluation_metrics.hpp
	$(CXX) $(CXXFLAGS) -o $@ main_nonstationary.cpp

clean:
	rm -f main_stationary main_nonstationary

.PHONY: all clean
