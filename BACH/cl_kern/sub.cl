void kernel sub(global const double *S, global const double *I,
                global double *D, const double invNorm) {
  int id = get_global_id(0);

  D[id] = (I[id] - S[id]) * -invNorm;
  // D[id] = S[id];
}