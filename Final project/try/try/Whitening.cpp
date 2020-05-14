//
//  Whitening.cpp
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#include <stdio.h>
#include "Whitening.h"
//two other whitening  method
/* this fuction is to make the brightness
 and contractness change linearly */
void whiteFace(Mat& matSelfPhoto,int alpha, int beta)
{
    Mat matResult;
    Mat matFinal;
    //change by linear method
    for (int y = 0; y < matSelfPhoto.rows; y++)
    {
        for (int x = 0; x < matSelfPhoto.cols; x++)
        {            for (int c = 0; c < 3; c++)
            {
                matSelfPhoto.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(matSelfPhoto.at<Vec3b>(y, x)[c]) + beta);
            }
        }
    }
}

/**
 Display a picture by gamma (non-linear）
 */
void gammaProcessImage(Mat& oriMat,double gamma,Mat outputMat){
    //The gamma method also modifies each pixel value according to a formula. We can write it through the LUT function. Its formula is:
    //O=(I/255)^×255
    Mat lookupTable(1,256,CV_8U);
    uchar* p = lookupTable.ptr();
    for (int i =0 ; i < 256; i++) {
        p[i] = saturate_cast<uchar>(pow(i/255.0, gamma) * 255.0);
    }
    Mat matSelfPhoto;
    outputMat = oriMat.clone();
    LUT(oriMat,lookupTable, outputMat);
    imshow("gg",outputMat);

}
/*this is the best whitening effect*/

/**
  Automatic brightness and contrast optimization with optional histogram clipping
  src Input image GRAY or BGR or BGRA
  dst Destination image
  clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
  In case of BGRA image, we won't touch the transparency
*/
void BrightnessAndContrastAuto(const Mat &src, Mat &dst, float clipHistPercent)
{
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, COLOR_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, COLOR_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;
    // alpha expands current range to histsize range
    beta = -minGray * alpha;
    // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3};
        mixChannels(&src, 4, &dst,1, from_to, 1);
    }
    return;
}
/*this is the best whitening effect*/

/**
  this function is to Adjust Brightness and Contrast
 */
int adjustBrightnessContrast(InputArray src, OutputArray dst, int brightness, int contrast)
{
    Mat input = src.getMat();
    if( input.empty() ) {
        return -1;
    }

    dst.create(src.size(), src.type());
    Mat output = dst.getMat();

    brightness = CLIP_RANGE(brightness, -255, 255);
    contrast = CLIP_RANGE(contrast, -255, 255);

    double B = brightness / 255.;
    double c = contrast / 255. ;
    double k = tan( (45 + 44 * c) / 180 * M_PI );

    Mat lookupTable(1, 256, CV_8U);
    uchar *p = lookupTable.data;
    for (int i = 0; i < 256; i++)
        p[i] = COLOR_RANGE( (i - 127.5 * (1 - B)) * k + 127.5 * (1 + B) );

    LUT(input, lookupTable, output);

    return 0;
}

