
COPTS = -std=c++20 -pedantic -Wall -Wextra -fcommon
LIBS  = -lCCfits -lcfitsio -lOpenCL
FILENAME = 

# compiler
CC    = g++

all: 
	$(CC) convolve.cc -o test $(LIBS) $(COPTS)

debug: 
	$(CC) convolve.cc -o test $(LIBS) -g $(COPTS)	

clean: 
	rm *.o
	rm test
