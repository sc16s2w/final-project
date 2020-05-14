//
//  Face thining.h
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#ifndef Face_thining_h
#define Face_thining_h
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

using namespace cv::face;
Mat face_thin(Mat &src,const vector<Point2f>& landmarks,int change,const Rect &face);
#endif /* Face_thining_h */
