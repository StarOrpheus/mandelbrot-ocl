struct Params {
  uint FrameWidth;
  double Scale;
  double2 Focus;
};

[[vk::push_constant]] ConstantBuffer<Params> Constants;

[[vk::binding(0, 0)]] RWStructuredBuffer<double> OutBuffer;

[[vk::constant_id(0)]] const uint MANDELBROT_ITERS_LIMIT = 64;

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint y = DTid.y;
    uint x = DTid.x;

    double c_real = x * Constants.Scale + Constants.Focus.x;
    double c_img = y * Constants.Scale + Constants.Focus.y;

    double z_real = 0;
    double z_img = 0;

    uint step = 0;
    while (true) {
      // multiplying z * z, then adding `c`
      double new_z_real = z_real * z_real - z_img * z_img + c_real;
      double new_z_img = 2 * z_real * z_img + c_img;

      z_real = new_z_real;
      z_img = new_z_img;
      ++step;

      if (z_real * z_real + z_img * z_img >= 4.) {
         OutBuffer[x * Constants.FrameWidth + y] = step / (double) MANDELBROT_ITERS_LIMIT;
         break;
      }

      if (step == MANDELBROT_ITERS_LIMIT) {
         OutBuffer[x * Constants.FrameWidth + y] = 0;
         break;
      }
    }
}
