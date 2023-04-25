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
void maskInput(Image& img);
void sigmaClip(std::vector<cl_double>& data, cl_double& mean, cl_double& stdDev,
               int iter);
bool inImage(Image& image, int x, int y);
void calcStats(Stamp& stamp, Image& image);

int ludcmp(std::vector<std::vector<cl_double>>& matrix, int matrixSize,
           std::vector<int>& index, cl_double& rowInter);
void lubksb(std::vector<std::vector<cl_double>>& matrix, int matrixSize,
            std::vector<int>& index, std::vector<cl_double>& result);
cl_double makeKernel(Kernel&, std::pair<cl_long, cl_long>, int x, int y);

/* SSS */
void createStamps(Image& img, std::vector<Stamp>& stamps, int w, int h);
double checkSStamp(SubStamp& sstamp, Image& image, Stamp& stamp);
cl_int findSStamps(Stamp& stamp, Image& image, int index);
int identifySStamps(std::vector<Stamp>& stamps, Image& image);

/* CMV */
void createB(Stamp& s, Image& img);
void convStamp(Stamp& s, Image& img, Kernel& k, int n, int odd);
void cutSStamp(SubStamp& ss, Image& img);
void fillStamp(Stamp& s, Image& tImg, Image& sImg, Kernel& k);

/* CD && KSC */
cl_double testFit(std::vector<Stamp>& stamps, Image& img);
void createMatrix(std::vector<Stamp>& stamps,
                  std::vector<std::vector<cl_double>>& matrix,
                  std::vector<std::vector<cl_double>>& weight,
                  std::pair<cl_long, cl_long>& imgSize);
void createScProd(std::vector<Stamp>& stamps, Image& img,
                  std::vector<std::vector<cl_double>>& weight,
                  std::vector<cl_double>& res);
cl_double calcSig(Stamp&, std::vector<cl_double>&, Image&);
cl_double getBackground(int x, int y, std::vector<cl_double>& kernSol,
                        std::pair<cl_long, cl_long> imgSize);
std::vector<cl_double> makeModel(Stamp& s, std::vector<cl_double>& kernSol,
                                 std::pair<cl_long, cl_long> imgSize);

#endif
