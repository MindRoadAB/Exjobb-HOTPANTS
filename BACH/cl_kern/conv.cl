void kernel conv(global const double *convKern, const long convWidth,
                 global const double *image, global double *outimg,
                 const long w, const long h) {
  int id = get_global_id(0);
  double acc = 0.0;
  long x = id % w;
  long y = id / w;

  int xSteps = ceil((double)w / (double)convWidth);
  int ySteps = ceil((double)h / (double)convWidth);

  int xS = x / convWidth;
  int yS = y / convWidth;

  int convOffset = (xS + (yS * xSteps)) * convWidth * convWidth;

  long lim = convWidth % 2 == 0 ? (convWidth / 2) : (convWidth / 2 + 1);
  for(long j = -(convWidth / 2); j < lim; j++) {
    for(long i = -(convWidth / 2); i < lim; i++) {
      if((x + i >= 0) && (x + i < w) && (y + j >= 0) && (y + j < h)) {
        long convIndex = -i + convWidth / 2 + (-j + convWidth / 2) * convWidth;
        convIndex += convOffset;
        long imgIndex = x + i + w * (j + y);
        acc += convKern[convIndex] * image[imgIndex];
      }
    }
  }

  outimg[id] = acc;
}
