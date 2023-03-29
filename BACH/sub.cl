void kernel sub(global const double *S, global const double *I,
                global double *D) {
  int id = get_global_id(0);

  D[id] = S[id] - I[id];
}