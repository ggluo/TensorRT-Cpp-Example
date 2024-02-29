CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wno-deprecated -Wfloat-conversion
DEBUGFLAGS = -DDEBUG
LDFLAGS = -L/path/to/TensorRT/lib
LDLIBS = -lnvinfer -lnvonnxparser -lcudart -lnvinfer_plugin

# Add the directory containing NvInfer.h to the include path
INCLUDEDIRS = -I/path/to/TensorRT/include -Isrc

SRCDIR = src
BINDIR = bin

SOURCES = $(wildcard $(SRCDIR)/*.cpp) main.cpp
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)
EXECUTABLE = main

.PHONY: all clean

all: $(EXECUTABLE)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) $(INCLUDEDIRS) -c $< -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(INCLUDEDIRS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR) $(EXECUTABLE)
