CC=g++
CFLAGS=$(shell pkg-config opencv --cflags) -g -Wall -Wconversion  -O3
CPPFLAGS=$(CFLAGS)
LDFLAGS=$(shell pkg-config opencv --libs) -lm -O3

all: astich
astich: astich.o

astich.o: astich.cpp

.PHONY: clean
clean:
	rm -f astich *.o
