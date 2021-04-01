#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cassert>
#include <limits.h>
#include <float.h>

#include "kalman.hpp"
#include "obj.hpp"
#include "utils.cpp"

#define NI 300                   // 仿真迭代次数
#define TI 0.1                   // 时间间隔
#define NT 30                    // 目标数量
#define ARANGE 20                // 量测范围
#define GATE_THRESHOLD 2000      // 跟踪门阈值
#define BLIND_UPDATE_LIMIT 5     // 中断更新的次数限制

#define SIGMA_AX 1               // 过程噪声标准差
#define SIGMA_AY 1               // 过程噪声标准差
#define SIGMA_OX 0.1             // 量测噪声标准差
#define SIGMA_OY 0.1             // 量测噪声标准差

int main()
{
    // 设置显示界面
    cv::namedWindow("Simulation of MTT", cv::WINDOW_AUTOSIZE);
    
    // 设置颜色
    std::vector<int> COLOR_B;
    COLOR_B = {244, 233, 156, 103, 63, 33, 3, 0, 0};
    std::vector<int> COLOR_G;
    COLOR_G = {67, 30, 99, 58, 81, 150, 169, 188, 150};
    std::vector<int> COLOR_R;
    COLOR_R = {54, 99, 176, 183, 181, 243, 244, 212, 136};

    assert((COLOR_B.size() == COLOR_G.size()) && (COLOR_B.size() == COLOR_R.size()));
    int num_c = COLOR_B.size();

    // 设置自车位置
    Eigen::Matrix<double, 2, 1> xtrue;
    xtrue << 0, 0;

    // 初始化跟踪列表
    std::vector<Object> objs;
    int number = 0;

    // 初始化临时跟踪列表
    std::vector<Object> objs_temp;

    // 初始化目标位置和速度
    srand(time(NULL));
    double randf = rand() / double(RAND_MAX);
    Eigen::Matrix<double, 4 * NT, 1> targets;
    for (int t = 0; t < NT; t ++)
    {
        targets(4 * t + 0, 0) = fabs(100 * generate_gaussian_noise(0, 1)) + 20;
        targets(4 * t + 1, 0) = 5 * generate_gaussian_noise(0, 1) - 5;
        targets(4 * t + 2, 0) = 10 * generate_gaussian_noise(0, 1);
        targets(4 * t + 3, 0) = 1 * generate_gaussian_noise(0, 1);
    }

    // 仿真迭代
    for (int i = 1; i < NI; i ++)
    {
        std::cout << std::endl;
        std::cout << "\nIteration:" << i << std::endl;
        int num;

        // 清空显示界面
        cv::Mat img(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));

        // 控制目标随机移动
        for (int j = 0; j < NT; j ++)
        {
            Eigen::Matrix<double, 4, 1> tar;
            tar << targets(4 * j + 0, 0), targets(4 * j + 1, 0), targets(4 * j + 2, 0), targets(4 * j + 3, 0);
            control_target(tar, TI, SIGMA_AX, SIGMA_AY);
            targets(4 * j + 0, 0) = tar(0, 0);
            targets(4 * j + 1, 0) = tar(1, 0);
            targets(4 * j + 2, 0) = tar(2, 0);
            targets(4 * j + 3, 0) = tar(3, 0);
        }

        // 初始化量测列表
        std::vector<Object> objs_observed;
        std::vector<Object> objs_observed_copy;

        // 获取量测
        for (int j = 0; j < NT; j ++)
        {
            Eigen::Matrix<double, 4, 1> tar;
            tar << targets(4 * j + 0, 0), targets(4 * j + 1, 0), targets(4 * j + 2, 0), targets(4 * j + 3, 0);
            Eigen::Matrix<double, 2, 1> z;
            bool flag = false;
            observe(tar, ARANGE, SIGMA_OX, SIGMA_OY, z, flag);

            if (flag)
            {
                Object obj;
                obj.xref = z(0, 0);
                obj.yref = z(1, 0);
                objs_observed.push_back(obj);
            }
        }
        objs_observed_copy.assign(objs_observed.begin(), objs_observed.end());

        // 数据关联与跟踪
        num = objs.size();
        for (int j = 0; j < num; j ++)
        {
            bool flag = false;
            int idx = 0;
            double ddm = DBL_MAX;

            int n = objs_observed.size();
            for (int k = 0; k < n; k ++)
            {
                double x = objs_observed[k].xref;
                double y = objs_observed[k].yref;
                double dd = objs[j].tracker.compute_the_residual(x, y);
                if ((dd < ddm) && (dd < GATE_THRESHOLD)) {idx = k; ddm = dd; flag = true;}
            }

            if (flag)
            {
                double zx = objs_observed[idx].xref;
                double zy = objs_observed[idx].yref;
                objs[j].tracker.predict();
                objs[j].tracker.update(zx, zy);
                objs[j].tracker_blind_update = 0;
                objs_observed.erase(objs_observed.begin() + idx);
            }
            else
            {
                objs[j].tracker.predict();
                objs[j].tracker_blind_update += 1;
            }
        }

        // 删除长时间未关联的目标
        std::vector<Object> objs_remained;
        num = objs.size();
        for (int j = 0; j < num; j ++)
        {
            if (objs[j].tracker_blind_update <= BLIND_UPDATE_LIMIT) {objs_remained.push_back(objs[j]);}
        }
        objs = objs_remained;

        // 增广跟踪列表
        num = objs_temp.size();
        for (int j = 0; j< num; j ++)
        {
            bool flag = false;
            int idx = 0;
            double ddm = DBL_MAX;

            int n = objs_observed.size();
            for (int k = 0; k < n; k ++)
            {
                double x = objs_observed[k].xref;
                double y = objs_observed[k].yref;
                double dd = objs_temp[j].tracker.compute_the_residual(x, y);
                if ((dd < ddm) && (dd < GATE_THRESHOLD)) {idx = k; ddm = dd; flag = true;}
            }

            if (flag)
            {
                double zx = objs_observed[idx].xref;
                double zy = objs_observed[idx].yref;
                double x = objs_temp[j].tracker.get_state()(0, 0);
                double y = objs_temp[j].tracker.get_state()(2, 0);
                objs_temp[j].tracker.init(TI, zx, (zx - x) / TI, zy, (zy - y) / TI, SIGMA_AX, SIGMA_AY, SIGMA_OX, SIGMA_OY);

                objs_observed.erase(objs_observed.begin() + idx);
                number += 1;
                objs_temp[j].number = number;
                objs_temp[j].color_r = COLOR_R[number % num_c];
                objs_temp[j].color_g = COLOR_G[number % num_c];
                objs_temp[j].color_b = COLOR_B[number % num_c];
                objs.push_back(objs_temp[j]);
            }
        }

        // 增广临时跟踪列表
        objs_temp = objs_observed;
        num = objs_temp.size();
        for (int j = 0; j < num; j ++)
        {
            double x = objs_temp[j].xref;
            double y = objs_temp[j].yref;
            objs_temp[j].tracker.init(TI, x, 0, y, 0, SIGMA_AX, SIGMA_AY, SIGMA_OX, SIGMA_OY);
        }

        // 绘制自车
        cv::Scalar color_ego(0, 0, 0);
        draw_ego_vehicle(img, xtrue, color_ego);

        // 绘制观测范围
        cv::Scalar color_arange(0, 0, 0);
        draw_arange(img, xtrue, ARANGE, color_arange);

        // 绘制真实目标
        cv::Scalar color_target(0, 0, 0);
        draw_targets(img, targets, color_target);

        // 绘制量测结果
        cv::Scalar color_observation(0, 0, 0);
        num = objs_observed_copy.size();
        for (int j = 0; j < num; j ++)
        {
            double zx = objs_observed_copy[j].xref;
            double zy = objs_observed_copy[j].yref;
            draw_observation(img, xtrue, zx, zy, color_observation);
        }
        
        // 绘制跟踪结果及跟踪门
        num = objs.size();
        for (int j = 0; j < num; j++)
        {
            Eigen::Matrix<double, 4, 1> xx = objs[j].tracker.get_state();
            cv::Scalar color_xx(objs[j].color_b, objs[j].color_g, objs[j].color_r);
            draw_targets(img, xx, color_xx);

            Eigen::Matrix<double, 2, 1> ab = objs[j].tracker.compute_association_gate(GATE_THRESHOLD);
            draw_gate(img, xx, ab(0, 0), ab(1, 0), color_xx);
        }

        imshow("Simulation of MTT", img);

        // 按Esc终止仿真程序
        // 等待按键事件int cv::waitKey(int delay=0)的返回值为按键的ASCII码，delay为延迟的毫秒数，delay<=0时无限延迟
        if (cv::waitKey(100) == 27) {break;}

    }

    cv::destroyWindow("Simulation of MTT");
    std::cout << "\nSimulation process finished!" << std::endl;

    return 0;
}



