CC=g++
CFLAGS=$(shell pkg-config opencv --cflags) -g -Wall -Wconversion  -O3
CPPFLAGS=$(CFLAGS)
FLANN_ROOT=../ext/flann/build/dst/lib64
LDFLAGS=$(shell pkg-config opencv --libs) -lm -O3 -L ${FLANN_ROOT} -lflann_cpp

all: astich
astich: astich.o

astich.o: astich.cpp

.PHONY: clean
clean:
	rm -f astich *.o
