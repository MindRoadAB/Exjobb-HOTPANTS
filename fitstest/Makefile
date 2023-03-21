COPTS = -std=c++20 -pedantic -Wall -Wextra -fcommon
LIBS  = -lCCfits -lcfitsio

# compiler
CC    = g++

all: 
	rm res/out_*
	$(CC) main.cpp -o test $(LIBS) $(COPTS)

debug: 
	$(CC) main.cpp -o test $(LIBS) -g $(COPTS)	

clean: 
	rm *.o
	rm test