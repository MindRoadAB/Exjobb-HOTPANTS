COPTS = -std=c++20 -pedantic -Wall -Wextra -fcommon
LIBS  = -lCCfits -lcfitsio -lOpenCL

FILES =  main.cpp bachUtil.cpp
BIN =   main.o bachUtil.o sssUtil.o cmvUtil.o


# compiler
CC    = g++

all: $(BIN)
	$(CC) $(COPTS) $(LIBS) -o BACH $(BIN)

debug: $(BIN)
	$(CC) $(COPTS) -g $(LIBS) -o BACH $(BIN)

main.o: main.cpp
	$(CC) $(COPTS) -g $(LIBS) -c main.cpp

bachUtil.o: bachUtil.cpp
	$(CC) $(COPTS) -g $(LIBS) -c bachUtil.cpp

sssUtil.o: sssUtil.cpp
	$(CC) $(COPTS) -g $(LIBS) -c sssUtil.cpp

cmvUtil.o: cmvUtil.o
	$(CC) $(COPTS) -g $(LIBS) -c cmvUtil.cpp

clean:
	rm *.o
	rm BACH
