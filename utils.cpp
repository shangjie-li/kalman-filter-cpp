#include <cassert>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

double generate_gaussian_noise(double mu, double sigma)
{
    const double epsilon = std::numeric_limits<double>::min();
    const double two_pi = 2.0*3.14159265358979323846;

    static double z0, z1;
    static bool generate;
    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    }
    while (u1 <= epsilon);

    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

void control_target(Eigen::Matrix<double, 4, 1> &tar,
                                  double ti,
                                  double sigma_ax,
                                  double sigma_ay)
{
    Eigen::Matrix<double, 2, 1> v;
    v(0, 0) = sigma_ax * generate_gaussian_noise(0, 1);
    v(1, 0) = sigma_ay * generate_gaussian_noise(0, 1);

    Eigen::Matrix<double, 4, 4> ff;
    ff << 1, ti, 0, 0, 0, 1, 0, 0, 0, 0, 1, ti, 0, 0, 0, 1;

    Eigen::Matrix<double, 4, 2> gg;
    gg << 0.5 * ti * ti, 0, ti, 0, 0, 0.5 * ti * ti, 0, ti;

    tar = ff * tar + gg * v;
}

void observe(Eigen::Matrix<double, 4, 1> tar,
             double arange,
             double sigma_ox,
             double sigma_oy,
             Eigen::Matrix<double, 2, 1> &z,
             bool &flag)
{
    Eigen::Matrix<double, 2, 4> hh;
    hh << 1, 0, 0, 0, 0, 0, 1, 0;

    double tx = tar(0, 0);
    double ty = tar(2, 0);
    if ((tx * tx + ty * ty) <= arange * arange)
    {
        Eigen::Matrix<double, 2, 1> w;
        w(0, 0) = sigma_ox * generate_gaussian_noise(0, 1);
        w(1, 0) = sigma_oy * generate_gaussian_noise(0, 1);

        z = hh * tar;
        z += w;
        flag = true;
    }
}

void transform_points(std::vector<double> xs,
                      std::vector<double> ys,
                      std::vector<int> &xst,
                      std::vector<int> &yst,
                      int height,
                      int width)
{
    assert(xs.size() == ys.size());
    int num = xs.size();

    xst.clear();
    yst.clear();

    for (int i = 0; i < num; i ++)
    {
        double x = xs[i];
        double y = ys[i];

        int xt = (int)(x * 10 + width / 2);
        int yt = (int)(- y * 10 + height / 2);

        xst.push_back(xt);
        yst.push_back(yt);
    }
}

void draw_ego_vehicle(cv::Mat &img,
                      Eigen::Matrix<double, 2, 1> p,
                      cv::Scalar color)
{
    cv::Size sz = img.size();
    int height = sz.height;
    int width = sz.width;
    
    std::vector<double> xs;
    std::vector<double> ys;
    xs = {p(0, 0)};
    ys = {p(1, 0)};

    std::vector<int> xst;
    std::vector<int> yst;
    transform_points(xs, ys, xst, yst, height, width);

    cv::Point p1(xst[0] - 5, yst[0] - 5);
    cv::Point p2(xst[0] + 5, yst[0] + 5);
    cv::rectangle(img, p1, p2, color, -1);
}

void draw_arange(cv::Mat &img,
                 Eigen::Matrix<double, 2, 1> p,
                 double arange,
                 cv::Scalar color)
{
    cv::Size sz = img.size();
    int height = sz.height;
    int width = sz.width;

    std::vector<double> xs;
    std::vector<double> ys;
    xs = {p(0, 0)};
    ys = {p(1, 0)};

    std::vector<int> xst;
    std::vector<int> yst;
    transform_points(xs, ys, xst, yst, height, width);

    int r = (int)(arange * 10);
    cv::Point p0(xst[0], yst[0]);
    cv::circle(img, p0, r, color, 1);
}

void draw_targets(cv::Mat &img,
                 Eigen::MatrixXd targets,
                 cv::Scalar color)
{
    cv::Size sz = img.size();
    int height = sz.height;
    int width = sz.width;

    int n = (int)(targets.rows() / 4);

    for (int i = 0; i < n; i ++)
    {
        std::vector<double> xs;
        std::vector<double> ys;
        xs = {targets(4 * i + 0, 0)};
        ys = {targets(4 * i + 2, 0)};

        std::vector<int> xst;
        std::vector<int> yst;
        transform_points(xs, ys, xst, yst, height, width);

        int r = 3;
        cv::Point p0(xst[0], yst[0]);
        cv::circle(img, p0, r, color, -1);
    }
}

void draw_observation(cv::Mat &img,
                      Eigen::Matrix<double, 2, 1> p,
                      double zx,
                      double zy,
                      cv::Scalar color)
{
    cv::Size sz = img.size();
    int height = sz.height;
    int width = sz.width;

    std::vector<double> xs;
    std::vector<double> ys;
    xs = {p(0, 0), zx};
    ys = {p(1, 0), zy};

    std::vector<int> xst;
    std::vector<int> yst;
    transform_points(xs, ys, xst, yst, height, width);

    cv::Point p1(xst[0], yst[0]);
    cv::Point p2(xst[1], yst[1]);
    cv::line(img, p1, p2, color);
}

void draw_gate(cv::Mat &img,
               Eigen::Matrix<double, 4, 1> xx,
               double a,
               double b,
               cv::Scalar color)
{
    cv::Size sz = img.size();
    int height = sz.height;
    int width = sz.width;
    
    std::vector<double> xs;
    std::vector<double> ys;
    xs = {xx(0, 0)};
    ys = {xx(2, 0)};

    std::vector<int> xst;
    std::vector<int> yst;
    transform_points(xs, ys, xst, yst, height, width);

    cv::Point p0(xst[0], yst[0]);
    cv::Size ab;
    ab.height = a;
    ab.width = b;
    cv::ellipse(img, p0, ab, 0, 0, 360, color);
}               

