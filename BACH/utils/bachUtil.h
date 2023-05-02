#ifndef BACH_UTIL
#define BACH_UTIL

#include <CL/opencl.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "argsUtil.h"
#include "datatypeUtil.h"

/* Utils */
void checkError(cl_int err);
void maskInput(Image& tImg, Image& sImg);
void sigmaClip(std::vector<cl_double>& data, cl_double& mean, cl_double& stdDev,
               int iter);
bool inImage(Image&, int x, int y);
void calcStats(Stamp&, Image&);

int ludcmp(std::vector<std::vector<cl_double>>& matrix, int matrixSize,
           std::vector<int>& index, cl_double& rowInter);
void lubksb(std::vector<std::vector<cl_double>>& matrix, int matrixSize,
            std::vector<int>& index, std::vector<cl_double>& result);
cl_double makeKernel(Kernel&, std::pair<cl_long, cl_long>, int x, int y);

/* SSS */
void createStamps(Image&, std::vector<Stamp>& stamps, int w, int h);
double checkSStamp(SubStamp&, Image&, Stamp&);
cl_int findSStamps(Stamp&, Image&, int index);
int identifySStamps(std::vector<Stamp>& stamps, Image&);

/* CMV */
void createB(Stamp&, Image&);
void convStamp(Stamp&, Image&, Kernel&, int n, int odd);
void cutSStamp(SubStamp&, Image&);
int fillStamp(Stamp&, Image& tImg, Image& sImg, Kernel&);

/* CD && KSC */
cl_double testFit(std::vector<Stamp>& stamps, Image& tImg, Image& sImg);
std::pair<std::vector<std::vector<cl_double>>,
          std::vector<std::vector<cl_double>>>
createMatrix(std::vector<Stamp>& stamps, std::pair<cl_long, cl_long>& imgSize);
std::vector<cl_double> createScProd(
    std::vector<Stamp>& stamps, Image&,
    std::vector<std::vector<cl_double>>& weight);
cl_double calcSig(Stamp&, std::vector<cl_double>& kernSol, Image& tImg,
                  Image& sImg);
cl_double getBackground(int x, int y, std::vector<cl_double>& kernSol,
                        std::pair<cl_long, cl_long> imgSize);
std::vector<cl_double> makeModel(Stamp&, std::vector<cl_double>& kernSol,
                                 std::pair<cl_long, cl_long> imgSize);
void fitKernel(Kernel&, std::vector<Stamp>& stamps, Image& tImg, Image& sImg);
bool checkFitSolution(Kernel&, std::vector<Stamp>& stamps, Image& tImg,
                      Image& sImg);

#endif
