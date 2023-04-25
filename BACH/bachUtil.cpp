#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "utils/bachUtil.h"

void checkError(cl_int err) {
  if(err != 0) {
    std::cout << "Error encountered with error code: " << err << std::endl;
    exit(err);
  }
}

void maskInput(Image& img) {
  for(long y = 0; y < img.axis.second; y++) {
    for(long x = 0; x < img.axis.first; x++) {
      long index = x + y * img.axis.first;
      int borderSize = args.hSStampWidth + args.hKernelWidth;
      if(x < borderSize || x > img.axis.first - borderSize || y < borderSize ||
         y > img.axis.second - borderSize)
        img.maskPix(x, y, Image::edge);
      if(img[index] >= args.threshHigh || img[index] <= args.threshLow) {
        img.maskAroundPix(x, y, Image::badInput);
      }
      if(std::isnan(img[index])) {
        img.maskPix(x, y, Image::nan);
      }
    }
  }
}

bool inImage(Image& image, int x, int y) {
  return !(x < 0 || x > image.axis.first || y < 0 || y > image.axis.second);
}

void sigmaClip(std::vector<cl_double>& data, cl_double& mean, cl_double& stdDev,
               int iter) {
  /* Does sigma clipping on data to provide the mean and stdDev of said
   * data
   */
  if(data.empty()) {
    std::cout << "Cannot send in empty vector to Sigma Clip" << std::endl;
    exit(1);
  }

  size_t currNPoints = 0;
  size_t prevNPoints = data.size();
  std::vector<bool> intMask(data.size(), false);

  // Do three times or a stable solution has been found.
  for(int i = 0; (i < iter) && (currNPoints != prevNPoints); i++) {
    currNPoints = prevNPoints;
    mean = 0;
    stdDev = 0;

    for(size_t i = 0; i < data.size(); i++) {
      if(!intMask[i]) {
        mean += data[i];
        stdDev += data[i] * data[i];
      }
    }

    if(prevNPoints > 1) {
      mean = mean / prevNPoints;
      stdDev = stdDev - prevNPoints * mean * mean;
      stdDev = std::sqrt(stdDev / double(prevNPoints - 1));
    } else {
      std::cout << "prevNPoints is: " << prevNPoints
                << "Needs to be greater than 1" << std::endl;
      exit(1);
    }

    prevNPoints = 0;
    double invStdDev = 1.0 / stdDev;
    for(size_t i = 0; i < data.size(); i++) {
      if(!intMask[i]) {
        // Doing the sigmaClip
        if(abs(data[i] - mean) * invStdDev > args.sigClipAlpha) {
          intMask[i] = true;
        } else {
          prevNPoints++;
        }
      }
    }
  }
}

void calcStats(Stamp& stamp, Image& image) {
  /* Heavily taken from HOTPANTS which itself copied it from Gary Bernstein
   * Calculates important values of stamps for futher calculations.
   * TODO: Fix Masking, very not correct. Also mask bad pixels found in here.
   * TODO: Compare results on same stamp on this and old version.
   */

  cl_double median, lfwhm, sum;  // temp for now

  std::vector<cl_double> values{};
  std::vector<cl_int> bins(256, 0);

  cl_int nValues = 100;
  cl_double upProc = 0.9;
  cl_double midProc = 0.5;
  cl_int numPix = stamp.size.first * stamp.size.second;

  if(numPix < nValues) {
    std::cout << "Not enough pixels in a stamp" << std::endl;
    exit(1);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<cl_int> randGenX(0, stamp.size.first - 1);
  std::uniform_int_distribution<cl_int> randGenY(0, stamp.size.second - 1);

  // Stop after randomly having selected a pixel numPix times.
  for(int i = 0; (i < numPix) && (values.size() < size_t(nValues)); i++) {
    int randX = randGenX(gen);
    int randY = randGenY(gen);

    // Random pixel in stamp in stamp coords.
    cl_int indexS = randX + randY * stamp.size.first;

    // Random pixel in stamp in Image coords.
    cl_int xI = randX + stamp.coords.first;
    cl_int yI = randY + stamp.coords.second;

    if(image.masked(xI, yI, Image::badInput, Image::nan) || stamp[indexS] < 0) {
      continue;
    }

    values.push_back(stamp[indexS]);
  }

  std::sort(begin(values), end(values));

  // Width of a histogram bin.
  cl_double binSize = (values[(int)(upProc * values.size())] -
                       values[(int)(midProc * values.size())]) /
                      (cl_double)nValues;

  // Value of lowest bin.
  cl_double lowerBinVal =
      values[(int)(midProc * values.size())] - (128.0 * binSize);

  // Contains all good Pixels in the stamp, aka not masked.
  std::vector<cl_double> maskedStamp{};
  for(int x = 0; x < stamp.size.first; x++) {
    for(int y = 0; y < stamp.size.second; y++) {
      // Pixel in stamp in stamp coords.
      cl_int indexS = x + y * stamp.size.first;

      // Pixel in stamp in Image coords.
      cl_int xI = x + stamp.coords.first;
      cl_int yI = y + stamp.coords.second;

      if(!image.masked(xI, yI, Image::badInput, Image::nan) &&
         stamp[indexS] >= 0) {
        maskedStamp.push_back(stamp[indexS]);
      }
    }
  }

  // sigma clip of maskedStamp to get mean and sd.
  cl_double mean, stdDev, invStdDev;
  sigmaClip(maskedStamp, mean, stdDev, 3);
  invStdDev = 1.0 / stdDev;

  int attempts = 0;
  cl_int okCount = 0;
  cl_double sumBins = 0.0;
  cl_double sumExpect = 0.0;
  cl_double lower, upper;
  while(true) {
    if(attempts >= 5) {
      std::cout << "Creation of histogram unsuccessful after 5 attempts";
      return;
    }

    std::fill(bins.begin(), bins.end(), 0);
    okCount = 0;
    sum = 0.0;
    sumBins = 0.0;
    sumExpect = 0.0;
    for(int y = 0; y < stamp.size.second; y++) {
      for(int x = 0; x < stamp.size.first; x++) {
        // Pixel in stamp in stamp coords.
        cl_int indexS = x + y * stamp.size.first;

        // Pixel in stamp in Image coords.
        cl_int xI = x + stamp.coords.first;
        cl_int yI = y + stamp.coords.second;

        if(image.masked(xI, yI, Image::badInput, Image::nan) ||
           stamp[indexS] < 0) {
          continue;
        }

        if((std::abs(stamp[indexS] - mean) * invStdDev) > args.sigClipAlpha) {
          continue;
        }

        int index = std::clamp(
            (int)std::floor((stamp[indexS] - lowerBinVal) / binSize), 0, 255);

        bins[index]++;
        sum += abs(stamp[indexS]);
        okCount++;
      }
    }

    if(okCount == 0 || binSize == 0.0) {
      std::cout << "No good pixels or variation in pixels" << std::endl;
      exit(1);
    }

    cl_double maxDens = 0.0;
    int lowerIndex, upperIndex, maxIndex;
    for(lowerIndex = upperIndex = 1; upperIndex < 255;
        sumBins -= bins[lowerIndex++]) {
      while(sumBins < okCount / 10.0 && upperIndex < 255) {
        sumBins += bins[upperIndex++];
      }
      if(sumBins / (upperIndex - lowerIndex) > maxDens) {
        maxDens = sumBins / (upperIndex - lowerIndex);
        maxIndex = lowerIndex;
      }
    }
    if(maxIndex < 0 || maxIndex > 255) maxIndex = 0;

    sumBins = 0.0;
    for(int i = maxIndex; sumBins < okCount / 10.0 && i < 255; i++) {
      sumBins += bins[i];
      sumExpect += i * bins[i];
    }
    cl_double modeBin = sumExpect / sumBins + 0.5;
    stamp.stats.skyEst = lowerBinVal + binSize * (modeBin - 1.0);

    lower = okCount * 0.25;
    upper = okCount * 0.75;
    sumBins = 0.0;
    int i = 0;
    for(; sumBins < lower; sumBins += bins[i++])
      ;
    lower = i - (sumBins - lower) / bins[i - 1];
    for(; sumBins < upper; sumBins += bins[i++])
      ;
    upper = i - (sumBins - upper) / bins[i - 1];

    if(lower < 1.0 || upper > 255.0) {
      if(args.verbose) {
        std::cout << "Expanding bin size..." << std::endl;
      }
      lowerBinVal -= 128.0 * binSize;
      binSize *= 2;
      attempts++;
    } else if(upper - lower < 40.0) {
      if(args.verbose) {
        std::cout << "Shrinking bin size..." << std::endl;
      }
      binSize /= 3.0;
      lowerBinVal = stamp.stats.skyEst - 128.0 * binSize;
      attempts++;
    } else
      break;
  }
  stamp.stats.fwhm = binSize * (upper - lower) / args.iqRange;
  int i = 0;
  for(i = 0, sumBins = 0; sumBins < okCount / 2.0; sumBins += bins[i++])
    ;
  median = i - (sumBins - okCount / 2.0) / bins[i - 1];
  lfwhm = binSize * (median - lower) * 2.0 / args.iqRange;
  median = lowerBinVal + binSize * (median - 1.0);
}

int ludcmp(std::vector<std::vector<cl_double>>& matrix, int matrixSize,
           std::vector<int>& index, cl_double& d) {
  std::vector<cl_double> vv(matrixSize + 1, 0.0);
  int maxI{};
  cl_double temp2{};

  d = 1.0;

  // Calculate vv
  for(int i = 1; i <= matrixSize; i++) {
    double big = 0.0;
    for(int j = 1; j <= matrixSize; j++) {
      temp2 = fabs(matrix[i][j]);
      if(temp2 > big) big = temp2;
    }
    if(big == 0.0) {
      std::cout << " Numerical Recipies run error" << std::endl;
      return 1;
    }
    vv[i] = 1.0 / big;
  }

  // Do the rest
  for(int j = 1; j <= matrixSize; j++) {
    for(int i = 1; i < j; i++) {
      double sum = matrix[i][j];
      for(int k = 1; k < i; k++) {
        sum -= matrix[i][k] * matrix[k][j];
      }
      matrix[i][j] = sum;
    }
    double big = 0.0;
    for(int i = j; i <= matrixSize; i++) {
      double sum = matrix[i][j];
      for(int k = 1; k < j; k++) {
        sum -= matrix[i][k] * matrix[k][j];
      }
      matrix[i][j] = sum;
      double dum = vv[i] * fabs(sum);
      if(dum >= big) {
        big = dum;
        maxI = i;
      }
    }
    if(j != maxI) {
      for(int k = 1; k <= matrixSize; k++) {
        double dum = matrix[maxI][k];
        matrix[maxI][k] = matrix[j][k];
        matrix[j][k] = dum;
      }
      d = -d;
      vv[maxI] = vv[j];
    }
    index[j] = maxI;
    matrix[j][j] = matrix[j][j] == 0.0 ? 1.0e-20 : matrix[j][j];
    if(j != matrixSize) {
      double dum = 1.0 / matrix[j][j];
      for(int i = j + 1; i <= matrixSize; i++) {
        matrix[i][j] *= dum;
      }
    }
  }

  return 0;
}

void lubksb(std::vector<std::vector<cl_double>>& matrix, int matrixSize,
            std::vector<int>& index, std::vector<cl_double>& result) {
  int ii{};

  for(int i = 1; i <= matrixSize; i++) {
    int ip = index[i];
    double sum = result[ip];
    result[ip] = result[i];
    if(ii) {
      for(int j = ii; j <= i - 1; j++) {
        sum -= matrix[i][j] * result[j];
      }
    } else if(sum) {
      ii = i;
    }
    result[i] = sum;
  }

  for(int i = matrixSize; i >= 1; i--) {
    double sum = result[i];
    for(int j = i + 1; j <= matrixSize; j++) {
      sum -= matrix[i][j] * result[j];
    }
    result[i] = sum / matrix[i][i];
  }
}

cl_double testFit(std::vector<Stamp>& stamps, Image& img) {
  int nComp1 = args.nPSF - 1;
  int nComp2 = ((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2;
  int nBGComp = ((args.backgroundOrder + 1) * (args.backgroundOrder + 2)) / 2;
  int matSize = nComp1 * nComp2 + nBGComp + 1;
  int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;
  std::vector<cl_double> kernelSum(stamps.size(), 0.0);
  std::vector<int> index(nKernSolComp);  // Internal between ludcmp and lubksb.

  int count = 0;
  for(auto& s : stamps) {
    if(!s.subStamps.empty()) {
      double d;
      std::vector<cl_double> testVec(args.nPSF + 2, 0.0);
      std::vector<std::vector<cl_double>> testMat(
          args.nPSF + 2, std::vector<cl_double>(args.nPSF + 2, 0.0));
      bool nan = false;

      for(int i = 1; i <= args.nPSF + 1; i++) {
        testVec[i] = s.B[i];
        for(int j = 1; j <= i; j++) {
          testMat[i][j] = s.Q[i][j];
          testMat[j][i] = testMat[i][j];
          if(std::isnan(testMat[j][i])) nan = true;
        }
      }

      ludcmp(testMat, args.nPSF + 1, index, d);
      lubksb(testMat, args.nPSF + 1, index, testVec);

      if(std::isnan(testVec[1])) {
        s.stats.norm = 1e10;
      } else {
        s.stats.norm = testVec[1];
        kernelSum[count] = testVec[1];
        count++;
      }
    }
  }

  cl_double kernelMean, kernelStdev;
  sigmaClip(kernelSum, kernelMean, kernelStdev, 10);

  // normalise
  for(auto& s : stamps) {
    s.stats.diff = std::abs((s.stats.norm - kernelMean) / kernelStdev);
  }

  // global fit
  std::vector<Stamp> testStamps{};
  for(auto& s : stamps) {
    if(s.stats.diff < args.threshKernFit && !s.subStamps.empty())
      testStamps.push_back(s);
  }

  std::vector<std::vector<cl_double>> matrix(
      matSize + 1, std::vector<cl_double>(matSize + 1, 0.0));
  std::vector<std::vector<cl_double>> weight(
      stamps.size(), std::vector<cl_double>(nComp2, 0.0));
  std::vector<cl_double> testKernSol(nKernSolComp, 0.0);

  // do fit
  createMatrix(testStamps, matrix, weight, img.axis);
  createScProd(testStamps, img, weight, testKernSol);

  double d;
  ludcmp(matrix, matSize, index, d);
  lubksb(matrix, matSize, index, testKernSol);

  Kernel testKern{};
  testKern.solution = testKernSol;
  kernelMean = makeKernel(testKern, img.axis, 0, 0);

  // calc merit value
  std::vector<cl_double> merit(testStamps.size(), 0.0);
  cl_double sig{};
  count = 0;
  for(auto& ts : testStamps) {
    sig = calcSig(ts, testKern.solution, img);
    if(sig != -1 && sig <= 1e10) merit[count++] = sig;
  }
  cl_double meritMean, meritStdDev;
  sigmaClip(merit, meritMean, meritStdDev, 10);
  meritMean /= kernelMean;
  if(count > 0) return meritMean;
  return 666;
}

cl_double getBackground(int x, int y, std::vector<cl_double>& kernSol,
                        std::pair<cl_long, cl_long> imgSize) {
  int BGComp = (args.nPSF - 1) *
                   (((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2) +
               1;
  cl_double bg = 0.0;
  int xf = (x - 0.5 * imgSize.first) / (0.5 * imgSize.second);
  int yf = (y - 0.5 * imgSize.second) / (0.5 * imgSize.second);

  cl_double ax = 1.0;
  for(int i = 0, k = 1; i <= args.backgroundOrder; i++) {
    cl_double ay = 1.0;
    for(int j = 0; j <= args.backgroundOrder - i; j++) {
      bg += kernSol[BGComp + k++] * ax * ay;
      ay *= yf;
    }
    ax *= xf;
  }
  return bg;
}

cl_double calcSig(Stamp& s, std::vector<cl_double>& kernSol, Image& img) {
  if(s.subStamps.empty()) return -1.0;
  int ssx = s.subStamps[0].imageCoords.first;
  int ssy = s.subStamps[0].imageCoords.second;

  cl_double background = getBackground(ssx, ssy, kernSol, img.axis);
  std::vector<cl_double> tmp{makeModel(s, kernSol, img.axis)};

  int sigCount = 0;
  cl_double signal = 0.0;
  for(int y = 0; y < args.fSStampWidth; y++) {
    int absY = y - args.hSStampWidth + ssy;
    for(int x = 0; x < args.fSStampWidth; x++) {
      int absX = x - args.hSStampWidth + ssx;

      int intIndex = x + y * args.fSStampWidth;
      int absIndex = absX + absY * img.axis.first;
      cl_double tDat = tmp[intIndex];

      cl_double diff = tDat - img[absIndex] + background;
      if(img.masked(absX, absY, Image::badInput) ||
         std::abs(img[absIndex]) <= 1e-10) {
        continue;
      } else {
        tmp[intIndex] = diff;
      }
      if(std::isnan(tDat) || std::isnan(img[absIndex])) {
        img.maskPix(absX, absY, Image::badInput);
        img.maskPix(absX, absY, Image::nan);
        continue;
      }

      sigCount++;
      signal += diff * diff / (std::abs(img[absIndex]) * 2);
    }
  }
  if(sigCount > 0) {
    signal /= sigCount;
    if(signal >= 1e10) signal = -1;
  } else {
    signal = -1.0;
  }
  return signal;
}

void createScProd(std::vector<Stamp>& stamps, Image& img,
                  std::vector<std::vector<cl_double>>& weight,
                  std::vector<cl_double>& res) {
  int nComp1 = args.nPSF - 1;
  int nComp2 = ((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2;
  int nBGComp = ((args.backgroundOrder + 1) * (args.backgroundOrder + 2)) / 2;

  int sI = 0;
  for(auto& s : stamps) {
    if(s.subStamps.empty()) {
      sI++;
      continue;
    }
    auto [ssx, ssy] = s.subStamps[0].imageCoords;

    cl_double p0 = s.B[1];
    res[1] += p0;

    for(int i = 1; i < nComp1; i++) {
      p0 = s.B[i + 1];
      for(int j = 0; j < nComp2; j++) {
        int indx = (i - 1) * nComp2 + j + 1;
        res[indx + 1] += p0 * weight[sI][j];
      }
    }

    for(int bgIndex = 0; bgIndex < nBGComp; bgIndex++) {
      cl_double q = 0.0;
      for(int x = -args.hSStampWidth; x <= args.hSStampWidth; x++) {
        for(int y = -args.hSStampWidth; y <= args.hSStampWidth; y++) {
          int index = x + args.hSStampWidth +
                      args.fSStampWidth * (y + args.hSStampWidth);
          q += s.W[nComp1 + bgIndex + 1][index] *
               img[x + ssx + (y + ssy) * img.axis.first];
        }
      }
      res[nComp1 * nComp2 + bgIndex + 2] += q;
    }

    sI++;
  }
}

void createMatrix(std::vector<Stamp>& stamps,
                  std::vector<std::vector<cl_double>>& matrix,
                  std::vector<std::vector<cl_double>>& weight,
                  std::pair<cl_long, cl_long>& imgSize) {
  int nComp1 = args.nPSF - 1;
  int nComp2 = ((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2;  // = 6
  int nComp = nComp1 * nComp2;
  int nBGVectors =
      ((args.backgroundOrder + 1) * (args.backgroundOrder + 2)) / 2;  // = 3
  int mat_size = nComp + nBGVectors + 1;

  int pixStamp = args.fSStampWidth * args.fSStampWidth;
  int hPixX = imgSize.first / 2, hPixY = imgSize.second / 2;

  for(size_t st = 0; st < stamps.size(); st++) {
    Stamp& s = stamps[st];
    if(s.subStamps.empty()) continue;

    auto [xss, yss] = s.subStamps[0].imageCoords;

    double a1 = 1.0;
    for(int k = 0, i = 0; i <= int(args.kernelOrder); i++) {
      double a2 = 1.0;
      for(int j = 0; j <= int(args.kernelOrder) - i; j++) {
        weight[st][k++] = a1 * a2;
        a2 *= cl_double(yss - hPixY) / hPixY;
      }
      a1 *= cl_double(xss - hPixX) / hPixX;
    }

    for(int i = 0; i < nComp; i++) {
      int i1 = i / nComp2;
      int i2 = i - i1 * nComp2;
      for(int j = 0; j <= i; j++) {
        int j1 = i / nComp2;
        int j2 = i - j1 * nComp2;

        matrix[i + 2][j + 2] +=
            weight[st][i2] * weight[st][j2] * s.Q[i1 + 2][j1 + 2];
      }
    }

    matrix[1][1] += s.Q[1][1];
    for(int i = 0; i < nComp; i++) {
      int i1 = i / nComp2;
      int i2 = i - i1 * nComp2;
      matrix[i + 2][1] += weight[st][i2] * s.Q[i1 + 2][1];
    }

    for(int iBG = 0; iBG < nBGVectors; iBG++) {
      int i = nComp + iBG + 1;
      int iVecBG = nComp1 + iBG + 1;
      for(int i1 = 1; i1 < nComp1 + 1; i1++) {
        double p0 = 0.0;

        for(int k = 0; k < pixStamp; k++) {
          p0 += s.W[i1][k] * s.W[iVecBG][k];
        }

        for(int i2 = 0; i2 < nComp2; i2++) {
          int jj = (i1 - 1) * nComp2 + i2 + 1;
          matrix[i + 1][jj + 1] += p0 * weight[st][i2];
        }
      }

      double p0 = 0.0;
      for(int k = 0; k < pixStamp; k++) {
        p0 += s.W[0][k] * s.W[iVecBG][k];
      }
      matrix[i + 1][1] += p0;

      for(int jBG = 0; jBG <= iBG; jBG++) {
        double q = 0.0;
        for(int k = 0; k < pixStamp; k++) {
          q += s.W[iVecBG][k] * s.W[nComp1 + jBG + 1][k];
        }
        matrix[i + 1][nComp + jBG + 2] += q;
      }
    }
  }

  for(int i = 0; i < mat_size; i++) {
    for(int j = 0; j <= i; j++) {
      matrix[j + 1][i + 1] = matrix[i + 1][j + 1];
    }
  }
}

cl_double makeKernel(Kernel& kern, std::pair<cl_long, cl_long> imgSize, int x,
                     int y) {
  /*
   * Calculates the kernel for a certain pixel, need finished kernelSol.
   */

  int k = 2;
  std::vector<cl_double> kernCoeffs(args.nPSF, 0.0);
  std::pair<cl_long, cl_long> hImgAxis =
      std::make_pair(0.5 * imgSize.first, 0.5 * imgSize.second);

  for(int i = 1; i < args.nPSF; i++) {
    double aX = 1.0;
    for(int iX = 0; iX < -args.kernelOrder; iX++) {
      double aY = 1.0;
      for(int iY = 0; iY < -args.kernelOrder - iX; iY++) {
        kernCoeffs[i] += kern.solution[k++] * aX * aY;
        aY *= cl_double(y - hImgAxis.second) / hImgAxis.second;
      }
      aX *= cl_double(x - hImgAxis.first) / hImgAxis.first;
    }
  }
  kernCoeffs[0] = kern.solution[1];

  for(int i = 0; i < args.fKernelWidth * args.fKernelWidth; i++) {
    kern.currKernel[i] = 0.0;
  }

  cl_double sumKernel = 0.0;
  for(int i = 0; i < args.fKernelWidth * args.fKernelWidth; i++) {
    for(int psf = 0; psf < args.nPSF; psf++) {
      kern.currKernel[i] += kernCoeffs[psf] * kern.kernVec[psf][i];
    }
    sumKernel += kern.currKernel[i];
  }

  return sumKernel;
}

std::vector<cl_double> makeModel(Stamp& s, std::vector<cl_double>& kernSol,
                                 std::pair<cl_long, cl_long> imgSize) {
  std::vector<cl_double> model(args.fSStampWidth * args.fSStampWidth, 0.0);

  std::pair<cl_long, cl_long> hImgAxis =
      std::make_pair(0.5 * imgSize.first, 0.5 * imgSize.second);
  auto [xss, yss] = s.subStamps.front().imageCoords;

  for(int i = 0; i < args.fSStampWidth * args.fSStampWidth; i++) {
    model[i] = kernSol[1] * s.W[0][i];
  }

  for(int i = 1, k = 2; i < args.nPSF; i++) {
    double aX = 1.0, coeff = 0.0;
    for(int iX = 0; iX < -args.kernelOrder; iX++) {
      double aY = 1.0;
      for(int iY = 0; iY < -args.kernelOrder - iX; iY++) {
        coeff += kernSol[k++] * aX * aY;
        aY *= cl_double(yss - hImgAxis.second) / hImgAxis.second;
      }
      aX *= cl_double(xss - hImgAxis.first) / hImgAxis.first;
    }

    for(int j = 0; j < args.fSStampWidth * args.fSStampWidth; j++) {
      model[i] = coeff * s.W[i][j];
    }
  }

  return model;
}
