#define MAX_PIXELS (3000*3000)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int get_ppm_metadata(const char * fname, int * xpix, int * ypix, int * max) {
  char ftype[40];
  char ctype[40] = "P6";
  char line[80];
  int err;

  FILE * fp;
  err = 0;

  if (fname == NULL) fname = "\0";
  fp = fopen (fname, "rb");
  if (fp == NULL) {
    fprintf (stderr, "read_ppm failed to open %s: %s\n", fname,
	     strerror (err));
    return 1;
  }
  
  fgets(line, 80, fp);
  sscanf(line, "%s", ftype);

  while (fgets(line, 80, fp) && (line[0] == '#'));

  sscanf(line, "%d%d", xpix, ypix);
  fscanf(fp, "%d\n", max);

  if(*xpix * *ypix > MAX_PIXELS) {
     fprintf (stderr, "Image size is too big\n");
    return 4;
 };
  if (fclose (fp) == EOF) {
    perror ("close failed");
    return 3;
  }
  return 0;
}

int read_ppm (const char * fname, int * xpix, int * ypix, int * max, char * data) {
  char ftype[40];
  char ctype[40] = "P6";
  char line[80];
  int err;

  FILE * fp;
  err = 0;

  if (fname == NULL) fname = "\0";
  fp = fopen (fname, "rb");
  if (fp == NULL) {
    fprintf (stderr, "read_ppm failed to open %s: %s\n", fname,
	     strerror (err));
    return 1;
  }
  
  fgets(line, 80, fp);
  sscanf(line, "%s", ftype);

  while (fgets(line, 80, fp) && (line[0] == '#'));

  sscanf(line, "%d%d", xpix, ypix);
  fscanf(fp, "%d\n", max);

  if(*xpix * *ypix > MAX_PIXELS) {
     fprintf (stderr, "Image size is too big\n");
    return 4;
 };
  if (strncmp(ftype, ctype, 2) == 0) {
    size_t read = fread (data, sizeof (char), *xpix * *ypix * 3, fp);
    if (read != *xpix * *ypix * 3) {
      std::cout << read << "\n";
      std::cout << *xpix * *ypix * 3 << "\n";
      perror ("Read failed");
      return 2;
    }
  } else {
    fprintf (stderr, "Wrong file format: %s\n", ftype);
  }

  if (fclose (fp) == EOF) {
    perror ("close failed");
    return 3;
  }
  return 0;

}


int write_ppm (const char * fname, int xpix, int ypix, char * data) {

  FILE * fp;
  int err = 0;

  if (fname == NULL) fname = "\0";
  fp = fopen (fname, "wb");
  if (fp == NULL) {
    fprintf (stderr, "write_ppm failed to open %s: %s\n", fname,
	     strerror (err));
    return 1;
  }
  
  fprintf (fp, "P6\n");
  fprintf (fp, "%d %d 255\n", xpix, ypix);
  if (fwrite (data, sizeof (char), xpix*ypix*3, fp) != xpix*ypix*3) {
    perror ("Write failed");
    return 2;
  }
  if (fclose (fp) == EOF) {
    perror ("Close failed");
    return 3;
  }
  return 0;
}

