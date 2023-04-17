COPTS = -std=c++20 -pedantic -Wall -Wextra -fcommon
LIBS  = -lCCfits -lcfitsio -lOpenCL

FILES =  main.cpp bachUtil.cpp
BIN =   main.o bachUtil.o sssUtil.o cmvUtil.o


# compiler
CC    = g++

all: $(BIN)
	$(CC) $(COPTS) $(LIBS) -o BACH $(BIN)

main.o: main.cpp
	$(CC) $(COPTS) $(LIBS) -c main.cpp

bachUtil.o: bachUtil.cpp
	$(CC) $(COPTS) $(LIBS) -c bachUtil.cpp

sssUtil.o: sssUtil.cpp
	$(CC) $(COPTS) $(LIBS) -c sssUtil.cpp

cmvUtil.o: cmvUtil.o
	$(CC) $(COPTS) $(LIBS) -c cmvUtil.cpp

debug:
	$(CC) main.cpp -o BACH $(LIBS) -g $(COPTS)

clean:
	rm *.o
	rm BACH
