CC=g++
CFLAGS=$(shell pkg-config opencv --cflags) -g -Wall -Wconversion 
CPPFLAGS=$(CFLAGS)
LDFLAGS=$(shell pkg-config opencv --libs) -lm

all: main.o sift_lib.o 
	${CC} sift_lib.o main.o ${CPPLAGS} ${LDFLAGS} -o sift
main.o: main.cpp sift_lib.h
	${CC} main.cpp ${CPPLAGS} ${LDFLAGS} -c
sift.o: sift_lib.h sift_lib.cpp
	${CC} sift_lib.cpp ${CPPLAGS} ${LDFLAGS} -c
clean:
	rm -f sift *.o
