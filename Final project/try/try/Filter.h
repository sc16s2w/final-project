//
//  Filter.h
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#ifndef Filter_h
#define Filter_h
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

using namespace cv::face;
//---------------【全局变量声明部分】-------------------------
//--------------【全局函数声明部分】-------------------------
//轨迹条回调函数
void beFiltering(cv::Mat &image);
void CompImageHist(Mat &src, MatND &b_hist, MatND &g_hist, MatND &r_hist);
#endif /* Filter_h */
