COPTS = -std=c++20 -pedantic -Wall -Wextra -fcommon
LIBS  = -lCCfits -lcfitsio -lOpenCL

# compiler
CC    = g++

all:
	$(CC) convolve.cpp -o test $(LIBS) $(COPTS)

debug:
	$(CC) convolve.cpp -o test $(LIBS) -g $(COPTS)

clean:
	rm *.o
	rm test
