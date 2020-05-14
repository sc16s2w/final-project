//
//  Filter.cpp
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#include <stdio.h>
#include "Filter.h"
void CompImageHist(Mat &src, MatND &b_hist, MatND &g_hist, MatND &r_hist)
{
    // Split into 3 single-channel images (bgr)
    vector<Mat> rgb_planes;
    split(src, rgb_planes);

    // Set the number of bins and the value range
    int histSize = 255;
    float range[] = { 0, 255 };
    const float* histRange = { range };

    // Calculate histogram
    bool uniform = true;
    bool accumulate = false;
    calcHist(&rgb_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&rgb_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    //Histogram Normalization >> Range is [0, 1]
    normalize(r_hist, r_hist, 0, 1, NORM_MINMAX, -1/*, Mat()*/);
    normalize(g_hist, g_hist, 0, 1, NORM_MINMAX, -1/*, Mat()*/);
    normalize(b_hist, b_hist, 0, 1, NORM_MINMAX, -1/*, Mat()*/);
}
/*this function is the best filtering
  after experiments and search for report*/
void beFiltering(cv::Mat &image) {
    //use space to exchange time
    int table[] = { 0, 2, 5, 7, 10, 12, 14, 17, 19, 21,
        23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 52, 54, 56, 58, 59,
        61, 63, 64, 66, 68, 69, 71, 73, 74, 76,
        77, 79, 80, 82, 83, 85, 86, 88, 89, 90,
        92, 93, 95, 96, 97, 99, 100, 101, 103, 104,
        105, 106, 108, 109, 110, 111, 113, 114, 115, 116,
        117, 119, 120, 121, 122, 123, 124, 125, 127, 128,
        129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
        139, 141, 142, 143, 144, 145, 146, 147, 148, 148,
        149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
        159, 160, 161, 162, 162, 163, 164, 165, 166, 167,
        168, 169, 169, 170, 171, 172, 173, 174, 174, 175,
        176, 177, 178, 179, 179, 180, 181, 182, 183, 183,
        184, 185, 186, 186, 187, 188, 189, 189, 190, 191,
        192, 192, 193, 194, 195, 195, 196, 197, 198, 198,
        199, 200, 200, 201, 202, 202, 203, 204, 204, 205,
        206, 207, 207, 208, 209, 209, 210, 211, 211, 212,
        212, 213, 214, 214, 215, 216, 216, 217, 218, 218,
        219, 219, 220, 221, 221, 222, 223, 223, 224, 224,
        225, 226, 226, 227, 227, 228, 229, 229, 230, 230,
        231, 231, 232, 233, 233, 234, 234, 235, 235, 236,
        237, 237, 238, 238, 239, 239, 240, 240, 241, 242,
        242, 243, 243, 244, 244, 245, 245, 246, 246, 247,
        247, 248, 248, 249, 249, 250, 250, 251, 251, 252,
        252, 253, 254, 254, 255, 255 };
    //the three typical method is shown as follows:
    Mat img1 = imread("/Users/wangsiwei/Desktop/毕业设计图片/美白/8.jpg");
    Mat img2 = imread("/Users/wangsiwei/Desktop/毕业设计图片/美白/82.jpg");
    Mat img3 = imread("/Users/wangsiwei/Desktop/毕业设计图片/美白/14.jpg");
    MatND hist0[3], hist1[3], hist2[3], hist3[3];
    //Calculate image histogram
    CompImageHist(img1, hist1[0], hist1[1], hist1[2]);
    CompImageHist(img2, hist2[0], hist2[1], hist2[2]);
    CompImageHist(img3, hist3[0], hist3[1], hist3[2]);
    CompImageHist(image, hist0[0], hist0[1], hist0[2]);
    double sum1[4] = { 0.0 };
    double results1[4] = { 0.0 };
    double sum2[4] = { 0.0 };
    double results2[4] = { 0.0 };
    double sum3[4] = { 0.0 };
    double results3[4] = { 0.0 };
    // compare the input picture and the three typical picture
    for (int i = 0; i < 3; i++)
    {
        // The first is related, the second is chi-square, the third is intersection, and the fourth is Pap
        results1[0] = compareHist(hist0[i], hist1[i], HISTCMP_CORREL);
        results1[1] = compareHist(hist0[i], hist1[i], HISTCMP_CHISQR);
        results1[2] = compareHist(hist0[i], hist1[i], HISTCMP_INTERSECT);
        results1[3] = compareHist(hist0[i], hist1[i], HISTCMP_BHATTACHARYYA);
        sum1[0] += results1[0];
        sum1[1] += results1[1];
        sum1[2] += results1[2];
        sum1[3] += results1[3];
    }
    for (int i = 0; i < 3; i++)
    {
        //The first is related, the second is chi-square, the third is intersection, and the fourth is Pap
        results2[0] = compareHist(hist0[i], hist2[i], HISTCMP_CORREL);
        results2[1] = compareHist(hist0[i], hist2[i], HISTCMP_CHISQR);
        results2[2] = compareHist(hist0[i], hist2[i], HISTCMP_INTERSECT);
        results2[3] = compareHist(hist0[i], hist2[i], HISTCMP_BHATTACHARYYA);
        sum2[0] += results2[0];
        sum2[1] += results2[1];
        sum2[2] += results2[2];
        sum2[3] += results2[3];
    }
    for (int i = 0; i < 3; i++)
    {
        //The first is related, the second is chi-square, the third is intersection, and the fourth is Pap
        results3[0] = compareHist(hist0[i], hist3[i], HISTCMP_CORREL);
        results3[1] = compareHist(hist0[i], hist3[i], HISTCMP_CHISQR);
        results3[2] = compareHist(hist0[i], hist3[i], HISTCMP_INTERSECT);
        results3[3] = compareHist(hist0[i], hist3[i], HISTCMP_BHATTACHARYYA);
        sum3[0] += results3[0];
        sum3[1] += results3[1];
        sum3[2] += results3[2];
        sum3[3] += results3[3];
    }
    double result[3];
    result[0] = sum1[2]/3;
    result[1] = sum2[2]/3;
    result[2] = sum3[2]/3;
    int count;
    double highest;
    highest = result[0];
    double value1 = 0;
    double p = 0;
    for (count = 1; count < 3; count++)
    {
        if (result[count] > highest)
            highest = result[count];
    }
    if(highest == result[0]){
        value1 = 3.78412698;
        p = 45.65079365;
    }
    if(highest == result[1]){
        value1 = 2.25;
        p = 21.31944444;
    }
    if(highest == result[2]){
        value1 = 3.03846154;
        p = 31.63076923;
    }
    int value2 = 1;     //Determining the degree of microdermabrasion and detail
    int dx = value1 * 5;    //One of the bilateral filtering parameters
    double fc = value1 * 12.5; //One of the bilateral filtering parameters
    Mat temp1, temp2, temp3, temp4;
    //Bilateral filtering
    bilateralFilter(image, temp1, dx, fc, fc);
    temp2 = (temp1 - image + 128);

    //Gaussian blur
    GaussianBlur(temp2, temp3, cv::Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

    temp4 = image + 2 * temp3 - 255;
    Mat img = image.clone();

    image = (image*(100 - p) + temp4 * p) / 100;
                    for (int i = 0; i < image.rows; ++i) {
                        uchar* data = image.ptr<uchar>(i);
                        for (int j = 0; j < image.cols*image.channels(); ++j) {
                                data[j] = table[data[j]];
                            }
                        }
}
