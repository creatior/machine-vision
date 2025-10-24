using Emgu.CV;
using Emgu.CV.Structure;

namespace GaussianFilterSimple
{
    class Program
    {
        static void Main(string[] args)
        {
            Image<Gray, byte> src = new Image<Gray, byte>("mole2.jpg");

            Image<Gray, byte> blurred = ApplyGaussianFilter(src, size: 7, sigma: 20);

            CvInvoke.Imshow("Исходное", src);
            CvInvoke.Imshow("После фильтра Гаусса", blurred);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }

        static double[,] CreateGaussianKernel(int size, double sigma)
        {
            double[,] kernel = new double[size, size];
            int center = size / 2;
            double sum = 0.0;

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    double dx = x - center;
                    double dy = y - center;
                    kernel[y, x] = Math.Exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                    sum += kernel[y, x];
                }
            }

            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    kernel[y, x] /= sum;

            return kernel;
        }

        static Image<Gray, byte> ApplyGaussianFilter(Image<Gray, byte> src, int size, double sigma)
        {
            double[,] kernel = CreateGaussianKernel(size, sigma);
            int pad = size / 2;

            int width = src.Width;
            int height = src.Height;

            double[,] padded = new double[height + 2 * pad, width + 2 * pad];

            for (int y = 0; y < height + 2 * pad; y++)
            {
                for (int x = 0; x < width + 2 * pad; x++)
                {
                    int yy = y - pad;
                    int xx = x - pad;

                    if (yy < 0) yy = -yy - 1;
                    if (yy >= height) yy = 2 * height - yy - 1;
                    if (xx < 0) xx = -xx - 1;
                    if (xx >= width) xx = 2 * width - xx - 1;

                    padded[y, x] = src[yy, xx].Intensity;
                }
            }

            Image<Gray, byte> result = new Image<Gray, byte>(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < size; i++)
                        for (int j = 0; j < size; j++)
                            sum += padded[y + i, x + j] * kernel[i, j];

                    result[y, x] = new Gray(Math.Min(255, Math.Max(0, sum)));
                }
            }

            return result;
        }
    }
}
