//
//  Skin detection.cpp
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#include <stdio.h>
#include "Skin detection.h"
#include "Filter.h"
#include "Whitening.h"
//Traditional RGB pixel feature detection+Automatic whitening threshold setting
 /*
 RGB_detect human skin detection
 Return value Mat type, extract the detected skin picture (convex hull)
 src reference type, the result of the final cutout (white background)
 output_mask reference type, the location of human skin pixels, if there is skin, it is saved as 255, otherwise it is 0
 */
 Mat RGB_detect(Mat &src, Mat &output_mask)
 {
    //to load 5 typical whietning pictures
     Mat img1 = imread("21.jpg");
     Mat img2 = imread("22.jpg");
     Mat img3 = imread("8.jpg");
     Mat img4 = imread("13.jpg");
     Mat img5 = imread("26.jpg");
     MatND hist0[3], hist1[3], hist2[3], hist3[3],hist4[3],hist5[3];
     // calculate picture histogram
     CompImageHist(img1, hist1[0], hist1[1], hist1[2]);
     CompImageHist(img2, hist2[0], hist2[1], hist2[2]);
     CompImageHist(img3, hist3[0], hist3[1], hist3[2]);
     CompImageHist(img4, hist4[0], hist4[1], hist4[2]);
     CompImageHist(img5, hist5[0], hist5[1], hist5[2]);
     CompImageHist(src, hist0[0], hist0[1], hist0[2]);
     double sum1[4] = { 0.0 };
     double results1[4] = { 0.0 };
     double sum2[4] = { 0.0 };
     double results2[4] = { 0.0 };
     double sum3[4] = { 0.0 };
     double results3[4] = { 0.0 };
     double sum4[4] = { 0.0 };
     double results4[4] = { 0.0 };
     double sum5[4] = { 0.0 };
     double results5[4] = { 0.0 };
     for (int i = 0; i < 3; i++)
     {
         //The first is related, the second is chi-square, the third is intersection, and the fourth is Pap
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
     for (int i = 0; i < 3; i++)
     {
        //The first is related, the second is chi-square, the third is intersection, and the fourth is Pap
        results4[0] = compareHist(hist0[i], hist4[i], HISTCMP_CORREL);
        results4[1] = compareHist(hist0[i], hist4[i], HISTCMP_CHISQR);
        results4[2] = compareHist(hist0[i], hist4[i], HISTCMP_INTERSECT);
        results4[3] = compareHist(hist0[i], hist4[i], HISTCMP_BHATTACHARYYA);
        sum4[0] += results4[0];
        sum4[1] += results4[1];
        sum4[2] += results4[2];
        sum4[3] += results4[3];
     }
     for (int i = 0; i < 3; i++)
     {
        //The first is related, the second is chi-square, the third is intersection, and the fourth is Pap
        results5[0] = compareHist(hist0[i], hist5[i], HISTCMP_CORREL);
        results5[1] = compareHist(hist0[i], hist5[i], HISTCMP_CHISQR);
        results5[2] = compareHist(hist0[i], hist5[i], HISTCMP_INTERSECT);
        results4[3] = compareHist(hist0[i], hist5[i], HISTCMP_BHATTACHARYYA);
        sum5[0] += results5[0];
        sum5[1] += results5[1];
        sum5[2] += results5[2];
        sum5[3] += results5[3];
     }
     double result[5];
     result[0] = sum1[2]/3;
     result[1] = sum2[2]/3;
     result[2] = sum3[2]/3;
     result[3] = sum4[2]/3;
     result[4] = sum4[2]/3;
     int count;
     double highest;
     highest = result[0];
     double alpha = 0;
     double beta = 0;
     //to find the largest one
     for (count = 1; count < 3; count++)
     {
         if (result[count] > highest)
             highest = result[count];
     }
     if(highest == result[0]){
         alpha = 5.45471698;
         beta = 4.62207547;
     }
     if(highest == result[1]){
         alpha = 15.21481481;
         beta = 14.1222222;
     }
     if(highest == result[2]){
         alpha = 10.4078125;
         beta = 9.7328125;
     }
     if(highest == result[3]){
         alpha = 2.52884615;
         beta = 2.09769231;
     }
     if(highest == result[4]){
         alpha = 8.66111111;
         beta = 7.86527778;
     }
     //to whitening the skin-are and non-skin area by dfferent amplitude
     Mat try1 = src.clone();
     Mat src2;
     BrightnessAndContrastAuto(src,src2,alpha);
     Mat src4;
     BrightnessAndContrastAuto(try1,src4,beta);
     output_mask = Mat::zeros(src.size(), CV_8UC1);
     Mat output_mask1 = Mat::zeros(src.size(), CV_8UC1);
     //foloow the rgb fomular
     for (int i = 0; i < src2.rows; i++)
     {
         for (int j = 0; j < src2.cols; j++)
         {
             int r = src2.at<cv::Vec3b>(i, j)[2];
             int g = src2.at<cv::Vec3b>(i, j)[1];
             int b = src2.at<cv::Vec3b>(i, j)[0];
             if (r > 95 && g > 40 && b > 20 && abs(r - g) > 15 && r > g && r > b && (max(max(r, g), b) - min(min(r, g), b)) > 15)
             {
                 output_mask.at<uchar>(i, j) = 255;
             }
             else{
                 output_mask1.at<uchar>(i, j) = 255;
             }
         }
     }
     Mat detect;
     Mat detect1;
     Mat final;
     src2.copyTo(detect, output_mask);
     src4.copyTo(detect1, output_mask1);
     detect.copyTo(detect1, output_mask);
     output_mask = detect1.clone();
     return output_mask;

 }
//HSV color space H range screening method
/*
HSV color space H range screening method
Return value Mat type, extract the detected skin picture (convex hull)
src reference type, the result of the final cutout (white background)
output_mask reference type, the location of human skin pixels, if there is skin, it is saved as 255, otherwise it is 0
*/
Mat HSV_detector(Mat& src)
{
    Mat hsv_image;
    int h = 0;
    int s = 1;
    int v = 2;
    cvtColor(src, hsv_image, COLOR_BGR2HSV); //First convert to YCrCb space
    Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
    Mat struElmen = getStructuringElement(MORPH_RECT,
    Size(3, 3), cv::Point(-1, -1));
    //follow the judge function
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            uchar *p_mask = output_mask.ptr<uchar>(i, j);
            uchar *p_src = hsv_image.ptr<uchar>(i, j);
            if (p_src[h] >= 0 && p_src[h] <= 20 && p_src[s] >=48 && p_src[v] >=50)
            {
                p_mask[0] = 255;
            }
        }
    }
    //do findcontours which can make the effect more clear
    morphologyEx(output_mask, output_mask, MORPH_CLOSE, struElmen);
    //Define contour parameters
    vector< vector<cv::Point> > contours;
    vector< vector<cv::Point> > resContours;
    vector< Vec4i > hierarchy;
    //Connected domain lookup
    findContours(output_mask, contours, hierarchy,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //Filter pseudo contours
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (fabs(contourArea(Mat(contours[i]))) > 1000)
            resContours.push_back(contours[i]);
    }
    output_mask.setTo(0);
    //Draw outline
    drawContours(output_mask, resContours, -1,
                 Scalar(255, 0, 0), FILLED);
    Mat detect;
    src.copyTo(detect, output_mask);;
    return detect;
}

//Ellipse model detection
/*
eclipse human skin detection
Return value Mat type, extract the detected skin picture (convex hull)
src reference type, the result of the final cutout (white background)
output_mask reference type, the location of human skin pixels, if there is skin, it is saved as 255, otherwise it is 0
*/
Mat ellipse_detect(Mat& src)
{
    Mat img = src.clone();
    Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);
    //Use opencv's own ellipse generation function to first generate a skin color ellipse model
    ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
    Mat ycrcb_image;
    Mat struElmen = getStructuringElement(MORPH_RECT,
    Size(3, 3), cv::Point(-1, -1));
    Mat output_mask = Mat::zeros(img.size(), CV_8UC1);
    //First convert to YCrCb space
    cvtColor(img, ycrcb_image, COLOR_BGR2YCrCb);
    //Skin detection using elliptical skin model
    for (int i = 0; i < img.cols; i++)
        for (int j = 0; j < img.rows; j++)
        {
            Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)
                //If it falls within the ellipse area of the skin model, the point is the skin pixel
                output_mask.at<uchar>(j, i) = 255;
        }
    morphologyEx(output_mask, output_mask, cv::MORPH_CLOSE, struElmen);
    //Define contour parameters
    vector< vector<cv::Point> > contours;
    vector< vector<cv::Point> > resContours;
    vector< Vec4i > hierarchy;
    //Connected domain lookup
    findContours(output_mask, contours, hierarchy,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //Filter pseudo contours
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (fabs(contourArea(Mat(contours[i]))) > 1000)
            resContours.push_back(contours[i]);
    }
    output_mask.setTo(0);
    //Draw outline
    drawContours(output_mask, resContours, -1,
                 Scalar(255, 0, 0), FILLED);
    Mat detect;
    img.copyTo(detect,output_mask);  //Back to skin color map
    return detect;
}
//YCrCb color space Cr component + Otsu method
/*
YCrCb color space Cr component + Otsu method
Return value Mat type, extract the detected skin picture (convex hull)
src reference type, the result of the final cutout (white background)
output_mask reference type, the location of human skin pixels, if there is skin, it is saved as 255, otherwise it is 0
*/
Mat YCrCb_Otsu_detect(Mat& src)
{
    Mat struElmen = getStructuringElement(MORPH_RECT,
    Size(3, 3), cv::Point(-1, -1));
    Mat ycrcb_image;
    //First convert to YCrCb space
    cvtColor(src, ycrcb_image, COLOR_BGR2YCrCb);
    Mat detect;
    vector<Mat> channels;
    split(ycrcb_image, channels);
    Mat output_mask = channels[1];
    threshold(output_mask, output_mask, 0, 255, THRESH_BINARY | THRESH_OTSU);
    //Morphological closed operation
    morphologyEx(output_mask, output_mask, cv::MORPH_CLOSE, struElmen);
    //Define contour parameters
    vector< vector<cv::Point> > contours;
    vector< vector<cv::Point> > resContours;
    vector< Vec4i > hierarchy;
    //Connected domain lookup
    findContours(output_mask, contours, hierarchy,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //Filter pseudo contours
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (fabs(contourArea(Mat(contours[i]))) > 1000)
            resContours.push_back(contours[i]);
    }
    output_mask.setTo(0);
    //Draw outline
    drawContours(output_mask, resContours, -1,
                 Scalar(255, 0, 0), FILLED);
    src.copyTo(detect, output_mask);
    return detect;
}
//YCrCb color space Cr, Cb range screening method
/*
YCrCb color space Cr,Cb range srceening method
Return value Mat type, extract the detected skin picture (convex hull)
src reference type, the result of the final cutout (white background)
output_mask reference type, the location of human skin pixels, if there is skin, it is saved as 255, otherwise it is 0
*/
Mat YCrCb_detect(Mat & src)
{
    Mat try1 = src.clone();
    Mat ycrcb_image;
    int Cr = 1;
    int Cb = 2;
    //First convert to YCrCb space
    cvtColor(src, ycrcb_image, COLOR_BGR2YCrCb);
    Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
    Mat struElmen = getStructuringElement(MORPH_RECT,
    Size(3, 3), cv::Point(-1, -1));
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            uchar *p_mask = output_mask.ptr<uchar>(i, j);
            uchar *p_src = ycrcb_image.ptr<uchar>(i, j);
            if (p_src[Cr] >= 133 && p_src[Cr] <= 173 && p_src[Cb] >= 77 && p_src[Cb] <= 127)
            {
                p_mask[0] = 255;
            }
        }
    }
    morphologyEx(output_mask, output_mask, cv::MORPH_CLOSE, struElmen);
    //Define contour parameters
    vector< vector<cv::Point> > contours;
    vector< vector<cv::Point> > resContours;
    vector< Vec4i > hierarchy;
    //Connected domain lookup
    findContours(output_mask, contours, hierarchy,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //Filter pseudo contours
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (fabs(contourArea(Mat(contours[i]))) > 1000)
            resContours.push_back(contours[i]);
    }
    output_mask.setTo(0);
    //Draw outline
    drawContours(output_mask, resContours, -1,
                 Scalar(255, 0, 0), FILLED);
    Mat detect;
    src.copyTo(detect, output_mask);;
    return detect;

}
