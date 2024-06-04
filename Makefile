CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wno-deprecated -Wfloat-conversion
DEBUGFLAGS = -DDEBUG
LDFLAGS = -L/home/gluo/local_lib/TensorRT-8.6.1.6/lib
LDLIBS = -lnvinfer -lnvonnxparser -lcudart -lnvinfer_plugin

# Add the directory containing NvInfer.h to the include path
INCLUDEDIRS = -I/home/gluo/local_lib/TensorRT-8.6.1.6/include -Isrc

SRCDIR = src
BINDIR = bin

SOURCES = $(wildcard $(SRCDIR)/*.cpp) main.cpp
SOURCES_C = $(wildcard $(SRCDIR)/*.cpp) main.c
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)
OBJECTS_C = $(SOURCES_C:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)

SOURCES_G = $(wildcard $(SRCDIR)/*.cpp) gaussian_blur.cpp
OBJECTS_G = $(SOURCES_G:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)

EXECUTABLE = main
EXECUTABLE_C = main_c
EXECUTABLE_G = gaussian_blur

.PHONY: all clean

all: $(EXECUTABLE) $(EXECUTABLE_C) $(EXECUTABLE_G)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(INCLUDEDIRS) -c $< -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(INCLUDEDIRS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(EXECUTABLE_C): $(OBJECTS_C)
	$(CXX) $(INCLUDEDIRS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(EXECUTABLE_G): $(OBJECTS_G)
	$(CXX) $(INCLUDEDIRS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR) $(EXECUTABLE)
