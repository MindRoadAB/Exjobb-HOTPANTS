void xconv(int x, int pos, global const double *img, global const float *kern,
           int kernst, float acc, int w) {
  if (x > 0) {
    acc += kern[kernst] * img[pos - 1];
  }
  acc += kern[kernst + 1] * img[pos];
  if (x < w - 1) {
    acc += kern[kernst + 2] * img[pos + 1];
  }
}

void kernel conv(global const double *image, const int w, const int h, global double *outImage) {
  int id = get_global_id(0);

  outImage[id] = image[id];
}
