# compiler
CXX    = g++

CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -fcommon
LOADLIBES  = -lCCfits -lcfitsio -lOpenCL

FILES =  main.cpp bachUtil.cpp
BIN =   main.o bachUtil.o sssUtil.o cmvUtil.o

all: $(BIN)
	$(CXX) $(CXXFLAGS) $(LOADLIBES) -o BACH $(BIN)

debug: $(BIN)
	$(CXX) $(CXXFLAGS) -g $(LOADLIBES) -o BACH $(BIN)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -g $(LOADLIBES) -c main.cpp

bachUtil.o: bachUtil.cpp
	$(CXX) $(CXXFLAGS) -g $(LOADLIBES) -c bachUtil.cpp

sssUtil.o: sssUtil.cpp
	$(CXX) $(CXXFLAGS) -g $(LOADLIBES) -c sssUtil.cpp

cmvUtil.o: cmvUtil.cpp
	$(CXX) $(CXXFLAGS) -g $(LOADLIBES) -c cmvUtil.cpp

.PHONY: clean
clean:
	rm *.o BACH
