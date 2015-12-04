# Makefile for DNS-algorithm skeleton
# Made for Franklin
# For other machines, replace CC = cc with
# mpicc or other appropriate compiler

CC = cc
CFLAGS = -O3
LIBS = 

dnsmat: main.o
	$(CC) $(CFLAGS) -o dnsmat main.o $(LIBS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f dnsmat *.o

