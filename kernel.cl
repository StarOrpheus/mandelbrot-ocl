__kernel void mandelbrot(precise_t focus_x, precise_t focus_y,
                         precise_t scale,
                         uint const N, uint const M,
                         __global precise_t* result)
{
    // using y as 0-index and x as 1-index in global work is ~3 times faster, than vice versa
    uint y = get_global_id(0);
    uint x = get_global_id(1);

    precise_t c_real = x * scale + focus_x;
    precise_t c_img = y * scale + focus_y;

    precise_t z_real = 0;
    precise_t z_img = 0;

    uint step = 0;
    while (true) {
      // multiplying z * z, then adding `c`
      precise_t new_z_real = z_real * z_real - z_img * z_img + c_real;
      precise_t new_z_img = 2 * z_real * z_img + c_img;

      z_real = new_z_real;
      z_img = new_z_img;
      ++step;

      if (z_real * z_real + z_img * z_img >= 4.) {
         result[x * M + y] = step / (precise_t) MANDELBROT_ITERS_LIMIT;
         break;
      }

      if (step == MANDELBROT_ITERS_LIMIT) {
         result[x * M + y] = 0;
         break;
      }
    }
}
