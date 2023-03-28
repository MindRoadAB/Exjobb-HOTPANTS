void kernel conv(global const double *convKern, const long convWidth,
                 global const double *image, global double *outimg,
                 const long w, const long h) {
  int id = get_global_id(0);
  double acc = 0.0;
  long x = id % w;
  long y = id / w;

  long lim = convWidth % 2 == 0 ? (convWidth / 2) : (convWidth / 2 + 1);
  for(long i = -(convWidth / 2); i < lim; i++) {
    for(long j = -(convWidth / 2); j < lim; j++) {
      if((x + i >= 0) && (x + i < w) && (y + j >= 0) && (y + j < h)) {
        long convIndex = i + convWidth / 2 + (j + convWidth / 2) * convWidth;
        long imgIndex = id + i + w * j;
        acc += convKern[convIndex] * image[imgIndex];
      }
    }
  }

  outimg[id] = acc;
}
