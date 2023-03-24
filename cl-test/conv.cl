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

void kernel conv(global const float *convkern, global const double *image,
                 global double *outimg, const int w, const int h) {
  int id = get_global_id(0);
  double acc = 0;
  int x = id % w;
  int y = id / w;

  if (y > 0) {
    xconv(x, id - w, image, convkern, 0, acc, w);
  }
  if (y < h - 1) {
    xconv(x, id + w, image, convkern, 6, acc, w);
  }
  xconv(x, id, image, convkern, 3, acc, w);

  outimg[id] = clamp(acc, 0.0, 255.0);
}
