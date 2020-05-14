//
//  Whitening.h
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#ifndef Whitening_h
#define Whitening_h
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;
#define CLIP_RANGE(value, min, max)  ( (value) > (max) ? (max) : (((value) < (min)) ? (min) : (value)) )
#define COLOR_RANGE(value)  CLIP_RANGE(value, 0, 255)
void whiteFace(Mat& matSelfPhoto,int alpha, int beta);
void gammaProcessImage(Mat& oriMat,double gamma,Mat outputMat);
void BrightnessAndContrastAuto(const Mat &src, Mat &dst, float clipHistPercent=0);
int adjustBrightnessContrast(InputArray src, OutputArray dst, int brightness, int contrast);
#endif /* Whitening_h */
