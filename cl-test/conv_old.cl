void pixconv(double *acc, double kern, global const uchar *img, int pos) {
    acc[0] += kern * (float)convert_float(img[pos*3]);
    acc[1] += kern * (float)convert_float(img[pos*3+1]);
    acc[2] += kern * (float)convert_float(img[pos*3+2]);
}

void kernel conv(global const double *convKern, const int convWidth, global const uchar *image, global uchar *outimg, const int w, const int h, global int* C) {
  int id = get_global_id(0);
  double acc[3] = {0,0,0};
  int x = id % w;
  int y = id / w;
  int f = 255;

  int lim = convWidth % 2 == 0 ? (convWidth/2) : (convWidth/2 + 1);
  for(int i = -(convWidth/2); i < lim; i++){
    for (int j = -(convWidth/2); j < lim; j++){
      if ((x + i >= 0) && (x + i < w) && (y + j >= 0) && (y + j < h) ){
        int convIndex = i + convWidth/2 + (j + convWidth/2) * convWidth;
        int imgIndex = id + i + w * j;
        C[convIndex] = convIndex;
        pixconv(acc, convKern[convIndex], image, imgIndex);
      }
    }
  }

  outimg[id*3] = convert_uchar(clamp(acc[0], 0.0, 255.0));
  outimg[id*3+1] = convert_uchar(clamp(acc[1], 0.0, 255.0));
  outimg[id*3+2] = convert_uchar(clamp(acc[2], 0.0, 255.0));
}
