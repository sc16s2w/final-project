//
//  Skin detection.h
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#ifndef Skin_detection_h
#define Skin_detection_h
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

using namespace cv::face;
Mat RGB_detect(Mat &src, Mat &output_mask);
Mat HSV_detector(Mat& src);
Mat ellipse_detect(Mat& src);
Mat YCrCb_Otsu_detect(Mat& src);
Mat YCrCb_detect(Mat & src);
#endif /* Skin_detection_h */
