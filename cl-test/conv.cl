void pixconv(float *acc, float kern, global const uchar *img, int pos) {
    acc[0] += kern * (float)convert_float(img[pos*3]);
    acc[1] += kern * (float)convert_float(img[pos*3+1]);
    acc[2] += kern * (float)convert_float(img[pos*3+2]);
}

void xconv(int x, int pos, global const uchar *img, global const float *kern, int kernst, float *acc, int w) {
        if(x > 0) {
            pixconv(acc, kern[kernst], img, pos - 1);
        }
        pixconv(acc, kern[kernst+1], img, pos);
        if(x < w - 1) {
            pixconv(acc, kern[kernst+2], img, pos + 1);
        }
}

void kernel conv(global const float *convkern, global const uchar *image, global uchar *outimg, const int w, const int h, global int* C) {
    int id = get_global_id(0);
    float acc[3];
    int x = id % w;
    int y = id / w;
    int f = 255;

    acc[0] = 0;
    acc[1] = 0;
    acc[2] = 0;

    if(y > 0) {
        xconv(x, id - w, image, convkern, 0, acc, w);
    }
    if (y < h - 1) {
        xconv(x, id + w, image, convkern, 6, acc, w);
    }
    xconv(x, id, image, convkern, 3, acc, w);

    outimg[id*3] = convert_uchar(clamp(acc[0], 0.0f, 255.0f));
    outimg[id*3+1] = convert_uchar(clamp(acc[1], 0.0f, 255.0f));
    outimg[id*3+2] = convert_uchar(clamp(acc[2], 0.0f, 255.0f));
    if(true)
        C[get_global_id(0) % 10] = (int)acc[0];
}
