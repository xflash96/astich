CC=g++
CFLAGS=$(shell pkg-config opencv --cflags) -g -Wall -Wconversion  -O3
CPPFLAGS=$(CFLAGS)
LDFLAGS=$(shell pkg-config opencv --libs) -lm -O3

all: astich
astich: astich.o sift_lib.o

astich.o: astich.cpp

sift_lib.o: sift_lib.cpp sift_lib.h

.PHONY: install
install:
	rm -f feats.yml
	./astich  -f 16 ../test/boli_small/*

.PHONY: clean
clean:
	rm -f astich *.o feats.yml core.*
