//
//  Face detect.h
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#ifndef Face_detect_h
#define Face_detect_h
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

using namespace cv::face;
#define COLOR Scalar(255, 200,0)
void drawPolyline();
void drawLandmarks(Mat &im, vector<Point2f> &landmarks);
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);
#endif /* Face_detect_h */
