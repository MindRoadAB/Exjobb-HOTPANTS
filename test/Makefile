COPTS = -std=c++20 -pedantic -Wall -Wextra -fcommon
LIBS  = -lCCfits -lcfitsio

# compiler
CC    = g++

all: 
	$(CC) main.cpp -o test $(LIBS) $(COPTS)

clean: 
	rm *.o
	rm test