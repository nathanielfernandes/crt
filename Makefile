LIBS = ./libs

LODEPNG = $(LIBS)/lodepng

SRCS = main.cu $(LODEPNG)/lodepng.cpp
INCS = ./include

build:
	nvcc -o main $(SRCS) -I $(INCS)

clean:
	rm -f main

# clean, build, and run
run: build
	./main






