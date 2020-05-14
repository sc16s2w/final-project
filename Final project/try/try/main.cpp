//
//  main.cpp
//  try
//
//  Created by 王思为 on 2020/2/10.
//  Copyright © 2020 王思为. All rights reserved.
//

#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include"Morpher.h"
#include "Filter.h"
#include "Whitening.h"
#include "Skin detection.h"
#include "Face detect.h"
#include "Face thining.h"
using namespace cv;
using namespace std;
using namespace cv::face;
//Global variables
static Mat src2,src1,dst1;
Mat mouse1, mouse2;//
double S = 1;//Transform intensity
int rMax = 35;//Transform radius Mouse circle size
Point prePoint = Point(-1, -1);
Point endPoint = Point(-1, -1);//Real-time coordinates
Mat frame;
Mat src,dst;
Mat g_srcIamge, g_dstImage1, g_dstImage2, g_dstImage3, g_dstImage4, g_dstImage5, g_dstImage6;
int g_nBoxFilterValue = 1;//Box filter kernel value
int g_nMeanBlurValue = 1;//Mean filter kernel value
int g_nGaussianBlurValue = 1;//Gaussian filtering kernel value
int g_nMedianBlurValue = 1;//Median filter kernel value
int g_nBilateralFilterValue = 1;//Bilateral filtering kernel value
int g_nGuidedFilterValue = 1;//Guided filtering kernel value
const int g_nMaxVal = 18; //Default slider maximum
int g_value1; //the billiteral value
int g_p;  //the opacity
Mat g_srcImage;//the input
Mat img_adjust;//the output
static string window_name = "photo";
static int brightness = 255;
static int contrast = 255;
void waitESC();
static void callbackAdjust(int , void *);
//the adjust function is written in main function
//public function
//function is defiend here
static void on_BoxFilter(int, void*);//Box filter
static void on_MeanBlur(int, void*);//Mean filter
static void on_GaussianBlur(int, void*);//Gaussian filtering
static void on_MedianBlur(int, void*);//Median filter
static void on_BilateralFilter(int, void*);//Bilateral filtering
static void on_GuidedFilter(int, void*);//Guided filtering
void guidedFilter(Mat &srcMat, Mat &guidedMat, Mat &dstImage, int radius, double eps);//Guided filtering
void other_filters(string image_directory);
static void filterother(int, void *);
Mat warping(Mat &src, int rMax, Point prePoint, Point endPoint);
void onMouse(int event, int x, int y, int flags, void *ustc);
/*
   Use the scroll bar to control the parameter values of 6 filtering methods.
    Box filtering, mean filtering, Gaussian filtering, median filtering, bilateral filtering, guided filtering。
*/
 
#define WINDOWNAME "filter result"
/* this function
 is to adjust the box filter*/
static void on_BoxFilter(int, void*)
{
    boxFilter(g_srcIamge, g_dstImage1, -1, Size(g_nBoxFilterValue * 2 + 1, g_nBoxFilterValue * 2 + 1));
    cout << "\nThe current processing effect is [box filtering], and its kernel size is:" << g_nBoxFilterValue * 2 + 1 << endl;
    imshow(WINDOWNAME, g_dstImage1);
}
/* this function
is to adjust the mean filter*/
static void on_MeanBlur(int, void*)
{
    blur(g_srcIamge, g_dstImage2, Size(g_nMeanBlurValue * 2 + 1, g_nMeanBlurValue * 2 + 1), Point(-1, -1));
    cout << "\nThe current processing effect is [mean filtering], and its kernel size is:" << g_nMeanBlurValue * 2 + 1 << endl;
    imshow(WINDOWNAME, g_dstImage2);
}
/* this function
is to adjust the Guassian filter*/
static void on_GaussianBlur(int, void*)
{
    GaussianBlur(g_srcIamge, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
    cout << "\nThe current processing effect is [Gaussian filtering], and its kernel size is:" << g_nGaussianBlurValue * 2 + 1 << endl;
    imshow(WINDOWNAME, g_dstImage3);
}
/* this function
is to adjust the median filter*/
static void on_MedianBlur(int, void*)
{
    medianBlur(g_srcIamge, g_dstImage4, g_nMedianBlurValue * 2 + 1);
    cout << "\nThe current processing effect is [median filtering], and its kernel size is:" << g_nMedianBlurValue * 2 + 1 << endl;
    imshow(WINDOWNAME, g_dstImage4);
}
/* this function
is to adjust the bilateral filter*/
static void on_BilateralFilter(int, void*)
{
    bilateralFilter(g_srcIamge, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue * 2, g_nBilateralFilterValue / 2);
    cout << "\nThe current processing effect is [bilateral filtering], and its kernel size is:" << g_nBilateralFilterValue << endl;
    imshow(WINDOWNAME, g_dstImage5);
}
/* this function
is to adjust the guided filter*/
static void on_GuidedFilter(int, void*)
{
    vector<Mat> vSrcImage, vResultImage;
    //【1】对源图像进行通道分离，并对每个分通道进行导向滤波操作
    split(g_srcIamge, vSrcImage);
    for (int i = 0; i < 3; i++)
    {
        Mat tempImage;
        vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);//将分通道转换成浮点型数据
        Mat cloneImage = tempImage.clone();    //将tempImage复制一份到cloneImage
        Mat resultImage;
        guidedFilter(tempImage, cloneImage, resultImage, g_nGuidedFilterValue * 2 + 1, 0.01);//对分通道分别进行导向滤波
        vResultImage.push_back(resultImage);//将分通道导向滤波后的结果存放到vResultImage中
    }
    //【2】将分通道导向滤波后结果合并
    merge(vResultImage, g_dstImage6);
    cout << "\nThe current processing is [guided filtering], and its kernel size is:" << g_nGuidedFilterValue * 2 + 1 << endl;
    imshow(WINDOWNAME, g_dstImage6);
}

/* this function
is to achieve the guided filter*/
void guidedFilter(Mat &srcMat, Mat &guidedMat, Mat &dstImage, int radius, double eps)
{
    //Convert the source image information, expand the input to a 64-bit floating point type for later multiplication
    srcMat.convertTo(srcMat, CV_64FC1);
    guidedMat.convertTo(guidedMat, CV_64FC1);
    //Calculation of various means-
    Mat mean_p, mean_I, mean_Ip, mean_II;
    //Generate mean_p of the image to be filtered
    boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));
    //Generate mean image mean_I
    boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));
    //Generate cross-correlation means mean_Ip
    boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));
    //Generate guide image autocorrelation mean mean_II
    boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));
    //Calculate the correlation coefficient, calculate the covariance co of Ip and the variance var of I
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_II - mean_I.mul(mean_I);
    //Calculation of parameter coefficients a, b
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    //Calculate the mean of coefficients a and b
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
    boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
    //Generate output matrix
    dstImage = mean_a.mul(srcMat) + mean_b;
}
/*this function is to create track bar and adjust the */
void other_filters(string image_directory){
    g_srcIamge = imread(image_directory);
    //create window
    namedWindow("original window", 1);
    imshow("orignianl window", g_srcIamge);//display window
    //Create 6 sliding bars for filtering on the WINDOWNAME window
    namedWindow(WINDOWNAME);//create window
    createTrackbar("boxing filters", WINDOWNAME, &g_nBoxFilterValue, g_nMaxVal, on_BoxFilter);//Create a box filter trackbar
    on_BoxFilter(g_nBoxFilterValue, 0); //Callback function of track bar
    createTrackbar("Mean filtering", WINDOWNAME, &g_nMeanBlurValue, g_nMaxVal, on_MeanBlur);//Create mean filter trackbar
    on_MeanBlur(g_nMeanBlurValue, 0);
    createTrackbar("Gaussian filtering", WINDOWNAME, &g_nGaussianBlurValue, g_nMaxVal, on_GaussianBlur);//Create a Gaussian filter trackbar
    on_GaussianBlur(g_nGaussianBlurValue, 0);
    createTrackbar("Median filtering", WINDOWNAME, &g_nMedianBlurValue, g_nMaxVal, on_MedianBlur);//Create a median filter trackbar
    on_MedianBlur(g_nMedianBlurValue, 0);
    createTrackbar("Bilateral filtering", WINDOWNAME, &g_nBilateralFilterValue, g_nMaxVal, on_BilateralFilter);//Create bilateral filter trackbar
    on_BilateralFilter(g_nBilateralFilterValue, 0);
    createTrackbar("guided filter", WINDOWNAME, &g_nGuidedFilterValue, g_nMaxVal, on_GuidedFilter);//create guide filter trackbar
    on_GuidedFilter(g_nGuidedFilterValue, 0);
    //exit the function
    cout << "if you enter q, this function will destroyed\n" << endl;
    while (char(waitKey(1)) != 'q'){}
    string ch;
    cin>>ch;
    if(ch == "q"){
        destroyWindow(WINDOWNAME);
    }
}

static void filterother(int, void *);
/* this function is used to adjust
    the 6 filters effect*/
void adjust_filters(string image_directory){
    g_value1 = 1;
    g_p = 10;
    //
    g_srcImage= imread(image_directory);
    namedWindow("effect adjust", 1);
    //create trackbar
    createTrackbar("value1","effect adjust", &g_value1, 20, filterother);
    createTrackbar("opacity","effect adjust", &g_p, 80, filterother);
    //to call the recall function
    filterother(g_value1,0);
    filterother(g_p,0);
    cout << "if you enter q, this function will destroyed\n" << endl;
    while(char(waitKey(1)) != 'q') {}
    string ch;
    cin>>ch;
    if(ch == "q"){
        destroyWindow("effect adjust");
    }
}
/* this function is used to adjust
the best filter effect*/
static void filterother(int, void *) {
    //to use space to exchange time
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
    int value2 = 1;     //Determining the degree of microdermabrasion and detail
    int dx = g_value1 * 10;    //One of the bilateral filtering parameters
    double fc = g_value1 * 12.5; //One of the bilateral filtering parameters
    Mat temp1, temp2, temp3, temp4;
    //双边滤波
    img_adjust = g_srcImage.clone();
    bilateralFilter(img_adjust, temp1, dx, fc, fc);
    temp2 = (temp1 - img_adjust + 128);
    //高斯模糊
    GaussianBlur(temp2, temp3, cv::Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
    temp4 = img_adjust + 2 * temp3 - 255;
    img_adjust = (img_adjust*(100 - g_p) + temp4 * g_p) / 100;
                    for (int i = 0; i < img_adjust.rows; ++i) {
                        uchar* data = img_adjust.ptr<uchar>(i);
                        for (int j = 0; j < img_adjust.cols*img_adjust.channels(); ++j) {
                                data[j] = table[data[j]];
                            }
                        }
    imshow("effect adjust", img_adjust);
}
//face thining function
/*Manual whitening according to local deformation*/
Mat warping(Mat &src, int rMax, Point prePoint, Point endPoint)
{
    int cols = src.cols;
    int rows = src.rows;
    Mat mask = Mat(Size(cols, rows), CV_8UC1, Scalar(0, 0, 0));
    Mat result = Mat(Size(cols, rows), CV_8UC3, Scalar(0, 0, 0));

    double x = 0, y = 0;
    //to calculte the change between the start point
    for (int i = prePoint.y-rMax; i < prePoint.y + rMax;++i)
    {
        uchar  *data = mask.ptr<uchar>(i);
        for (int j= prePoint.x - rMax; j < prePoint.x + rMax; ++j)
        {

            double r = sqrt((i - prePoint.y)*(i - prePoint.y) + (j - prePoint.x)*(j - prePoint.x));
            if (r <= rMax && i >= 0 && j >= 0 && i < rows && j < cols)
            {

                data[j] = 255;
                double temp1 = r*r;
                double temp2 = 1.0*(endPoint.x - prePoint.x)* (endPoint.x - prePoint.x)
                    + 1.0*(endPoint.y - prePoint.y)* (endPoint.y - prePoint.y);

                double temp = 1.0*(rMax*rMax - temp1) / (rMax*rMax - temp1 + (100.0/ S)*temp2);

                x = j - (endPoint.x - prePoint.x)*temp*temp;
                y = i - (endPoint.y - prePoint.y)*temp*temp;

                int x1 = (int)x;
                int y1 = (int)y;
                int x2 = x1 + 1;
                int y2 = y1 + 1;

                Vec3b src1 = src.at<Vec3b>(y1, x1);
                Vec3b src2 = src.at<Vec3b>(y1, x2);
                Vec3b src3 = src.at<Vec3b>(y2, x1);
                Vec3b src4 = src.at<Vec3b>(y2, x2);

                Vec3d up,down;
                up[0] = (double)(src1[0] * (x - x1) + src2[0] * (x2 - x));
                up[1] = (double)(src1[1] * (x - x1) + src2[1] * (x2 - x));
                up[2] = (double)(src1[2] * (x - x1) + src2[2] * (x2 - x));


                down[0] = (double)(src3[0] * (x - x1) + src4[0] * (x2 - x));
                down[1] = (double)(src3[1] * (x - x1) + src4[1] * (x2 - x));
                down[2] = (double)(src3[2] * (x - x1) + src4[2] * (x2 - x));


                result.at<Vec3b>(i, j)[0] = (cvRound)(up[0]*(y - y1) + down[0]*(y2 - y));
                result.at<Vec3b>(i, j)[1] = (cvRound)(up[1] * (y - y1) + down[1] * (y2 - y));
                result.at<Vec3b>(i, j)[2] = (cvRound)(up[2] * (y - y1) + down[2] * (y2 - y));
            }
        }
    }

    for (int i = 0; i < rows;++i)
    {
        uchar * maskData = mask.ptr<uchar>(i);
        Vec3b * srcData = src.ptr<Vec3b>(i);
        Vec3b * dstData = result.ptr<Vec3b>(i);
        for (int j = 0; j < cols;++j)
        {
            if (maskData[j]==255)
                continue;
            dstData[j] = srcData[j];
        }
    }

    return result;
}
/* Mouse control for manual face thinning*/
void onMouse(int event, int x, int y, int flags, void *ustc)
{

    if (event == EVENT_LBUTTONDOWN)//Left click
    {
        mouse1 = src.clone();
        prePoint = Point(x, y);
        circle(mouse1, prePoint, rMax, Scalar(0, 255, 0, 0), 1, 8, 0);
        circle(mouse1, prePoint, 1, Scalar(255, 0, 255, 0), 1, 8, 0);
        imshow("src", mouse1);

    }

    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))//
    {
        mouse2 = mouse1.clone();
        endPoint = Point(x, y);
        if (endPoint.x < 0)endPoint.x = 0;
        if (endPoint.y < 0)endPoint.y = 0;
        if (endPoint.x >= mouse1.cols)endPoint.x = mouse1.cols - 1;
        if (endPoint.y >= mouse1.rows)endPoint.y = mouse1.rows - 1;
        circle(mouse2, endPoint, 1, Scalar(255, 0, 255, 0), 1, 8, 0);
        imshow("src", mouse2);

    }
    else if (event == EVENT_LBUTTONUP)
    {
        frame = mouse2.clone();
        endPoint = Point(x, y);
        if (endPoint.x < 0)endPoint.x = 0;
        if (endPoint.y < 0)endPoint.y = 0;
        if (endPoint.x >= frame.cols)endPoint.x = frame.cols-1;
        if (endPoint.y >= frame.rows)endPoint.y = frame.rows - 1;

        circle(frame, endPoint, 1, Scalar(255, 0, 255, 0), 1, 8, 0);

        line(frame, prePoint, endPoint, Scalar(255, 0, 255, 0), 1, 8, 0);

        imshow("src", frame);
        Mat result=warping(dst,rMax, prePoint, endPoint);
        dst = result.clone();
        imshow("dst", dst);
    }
}

/* this function is to use wait*/
void waitESC(){
    while(int key=waitKey(0) != 27){};
}
static void callbackAdjust(int , void *)
{
    Mat dst;
    adjustBrightnessContrast(src2, dst, brightness - 255, contrast - 255);
    imshow(window_name, dst);
}
//the main function
int main(){
    printf("+-----------------------------------------------------------------------+\n");
    printf("|----------------welcome to the beauty portrait tool--------------------|\n");
    printf("|--------------------1.face polishing function--------------------------|\n");
    printf("|--------------------2.face whitening function--------------------------|\n");
    printf("|--------------------3.face thinning function---------------------------|\n");
    printf("|--------------------4.auto make up function----------------------------|\n");
    printf("|--------------------5.other effect display-----------------------------|\n");
    printf("+-----------------------------------------------------------------------+\n");
    printf("------------------please enter the image directory-----------------------\n");
    printf("-the format is written as /Users/wangsiwei/Desktop/毕业设计图片/美白/171.jpg\n");
    string image_directory;
    cin>>image_directory;
    cout<<image_directory<<endl;
    Mat src_original = imread(image_directory);
    printf("----------now you can choose which function you want(1/2/3/4/5)------------\n");
    string choice;
    cin>>choice;
    //filter function
    if(choice == "1"){
        printf("+-----------------------------------------------------------------------+\n");
        printf("|----------------welcome to the face polishing function-----------------|\n");
        printf("|--------------------1.anto face polishing -----------------------------|\n");
        printf("|-------------2.display of other polishing filter effect----------------|\n");
        printf("|----------------3.adjust best face polishing algorithm-----------------|\n");
        printf("+-----------------------------------------------------------------------+\n");
        printf("------------------please enter your choice(1/2/3)-----------------------\n");
        printf("-the format is written as /Users/wangsiwei/Desktop/毕业设计图片/美白/171.jpg\n");
        string choice_1;
        cin>>choice_1;
        if(choice_1 == "1"){
            src = imread(image_directory);
            if (!src.data){
                cout << "NO DATA" << endl;
            }
            printf("the auto face polishing effect display\n");
            beFiltering(src);
            imshow("auto polishing effect display",src);
        }
        if(choice_1 == "2"){
            printf("other 6 face polishing effect display, you can adjust\n");
            other_filters(image_directory);
        }
        if(choice_1 == "3"){
            printf("adjust best face polishing algorithm\n");
            adjust_filters(image_directory);
        }
    }
    //whitening function
    if(choice == "2"){
        printf("+-----------------------------------------------------------------------+\n");
        printf("|----------------welcome to the face whitening function-----------------|\n");
        printf("|--------------------1.anto face whitening -----------------------------|\n");
        printf("|-------------2.display of other skin detection effect------------------|\n");
        printf("|----------------3.display of other two whitening effect----------------|\n");
        printf("|----------------4.adjust best whitening effct algorithm----------------|\n");
        printf("+-----------------------------------------------------------------------+\n");
        printf("------------------please enter your choice(1/2/3/4)---------------------\n");
        printf("-the format is written as /Users/wangsiwei/Desktop/毕业设计图片/美白/171.jpg\n");
        string choice_2;
        cin>>choice_2;
        if(choice_2 == "1"){
            printf("the auto face whitening effect display\n");
            Mat rgb_detect;
            RGB_detect(src_original, rgb_detect);
            imshow("rgb_detect",rgb_detect);
            printf("whether you want to save it or not\n");
            imwrite("/Users/wangsiwei/Desktop/毕业设计图片/美白/result.jpg",rgb_detect);
            }
        if(choice_2 == "2"){
            printf("display of other skin detection effect\n");
            Mat eclipse_detect;
            Mat ycrcb_detect;
            Mat ycrcb_improve_detect;
            Mat hsv_detect;
            Mat src_eclipse = imread(image_directory);
            if (!src_eclipse.data){
                cout << "NO DATA" << endl;
            }
            Mat src_ycrcb = imread(image_directory);
            if (!src_ycrcb.data){
                cout << "NO DATA" << endl;
            }
            Mat src_ycrcb_improve = imread(image_directory);
            if (!src_ycrcb_improve.data){
                cout << "NO DATA" << endl;
            }
            Mat src_hsv = imread(image_directory);
            if (!src_hsv.data){
                cout << "NO DATA" << endl;
            }
            eclipse_detect = ellipse_detect(src_eclipse);
            ycrcb_detect = YCrCb_detect(src_ycrcb);
            ycrcb_improve_detect = YCrCb_Otsu_detect(src_ycrcb_improve);
            hsv_detect = HSV_detector(src_hsv);
            imshow("eclipse detect",eclipse_detect);
            imshow("ycrcb detect",ycrcb_detect);
            imshow("ycrcb otu detect",ycrcb_improve_detect);
            imshow("hsv detect",hsv_detect);
            }
        if(choice_2 == "3"){
            printf("display of other two whitening effect\n");
            printf("the fist method is written as the linear change of brightness and contractness:\n");
            printf("now, you should input the alpha(1-2)\n");
            int alpha;
            int beta;
            cin>>alpha;
            printf("now, you should input the beta(20-50)\n");
            cin>>beta;
            src = imread(image_directory);
            if (!src.data){
                cout << "NO DATA" << endl;
            }
            whiteFace(src, alpha, beta);
            imshow("normal linear method", src);
            printf("the second method is written as the non-linear change of brightness and contractness:\n");
            printf("now, you can input the gamma(0-1)\n");
            double gamma;
            cin>>gamma;
            Mat gamma_origin = imread(image_directory);
            Mat gamma_result;
            gammaProcessImage(gamma_origin, gamma, gamma_result);
            imshow("gamma effect",gamma_result);
        }
        if(choice_2 == "4"){
            printf("the auto face whitening effect display\n");
            src1=imread(image_directory);
            if (!src1.data){
                cout << "NO DATA" << endl;
            }
            BrightnessAndContrastAuto(src1,src2,5);
            namedWindow(window_name);
            createTrackbar("brightness", window_name, &brightness, 2*brightness, callbackAdjust);
            createTrackbar("contrast", window_name, &contrast, 2*contrast, callbackAdjust);
            callbackAdjust(0, 0);
            waitESC();
        }
    }
    //thinning function
    if(choice == "3"){
           printf("+---------------------------------------------------------------------+\n");
           printf("|----------------welcome to the face thining function-----------------|\n");
           printf("|-----------------1.anto face thinning effect display ----------------|\n");
           printf("|-----------------2.other face thining effect-------------------------|\n");
           printf("|---------------------------------------------------------------------|\n");
           printf("+-----------------please enter your choice(1/2)-----------------------+\n");
           string choice_3;
           cin>>choice_3;
        if(choice_3 == "1"){
            printf("now you can see the different amplitude face thinging(5/10/15\n");
            Mat src = imread(image_directory);
            if (!src.data){
                cout << "NO DATA" << endl;
            }
            CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
            //Create an object of the Facemark class
            Ptr<Facemark> facemark = FacemarkLBF::create();
            // Load the face detector model
            facemark->loadModel("lbfmodel.yaml");
            //load the picture
            Mat frame, gray;
            frame = src.clone();
            // load frame
            vector<Rect> faces;
            //Container for storing rectangular frame of human face
            Rect face;
            //Convert frames to grayscale, because the input of Face Detector is grayscale
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            // face detector
            faceDetector.detectMultiScale(gray, faces);
            //Face key point container
            vector< vector<Point2f> > landmarks;
            vector<Point2f> landmark;
            vector<Point2f> landmar;
            // Run the face key detector (landmark detector)
            int i = 0;
            int j = 0;
            bool success = facemark->fit(frame,faces,landmarks);
            if(success)
            {
            //if success, draw the landmarks
            for(i = 0; i < landmarks.size(); i++)
            {
                for(j =0; j<landmarks[i].size();j++){
                    landmark.insert(landmark.end(),landmarks[i][j]);
                }
            }
            drawLandmarks(frame, landmark);// OpenCV comes with a key point drawing function: drawFacemarks
                drawFacemarks(frame, landmark, Scalar(0, 0, 255));
            Mat pic = face_thin(src, landmark, 5, faces[0]);//face thin on amplitude 5
            Mat pic1 = face_thin(src, landmark, 10, faces[0]);//face thin on amplitude 5
            Mat pic2 = face_thin(src, landmark, 15, faces[0]);//face thin on amplitude 5
            //display the effect
            imshow("Facial thin 5", pic);
            imshow("Facial thin 10", pic1);
            imshow("Facial thin 15", pic2);
        }
    }
        if(choice_3 == "2"){
            printf("here is the mannual face thining effect\n");
            printf("the left one is the original picture, and the right one is the changed picture\n");
            src= imread(image_directory,1);
            if (!src.data){
                cout << "NO DATA" << endl;
            }
            dst = src.clone();
            imshow("src", src);
            imshow("dst", dst);
            waitKey(100);
            while (1)
            {
                setMouseCallback("src", onMouse, 0);//Call the callback function
                waitKey(0);
            }
        }
    }
    //make up migration function
    if(choice == "4"){
           printf("+---------------------------------------------------------------------+\n");
           printf("|-------------welcome to the face make up migration function----------|\n");
           printf("|-----------------1.anto face make up migration ----------------------|\n");
           printf("|-----------------2.simple face morphing effect-----------------------|\n");
           printf("|---------------------------------------------------------------------|\n");
           printf("+-----------------please enter your choice(1/2)-----------------------+\n");
           string choice_4;
           cin>>choice_4;
        //beauty GAN effect display
        if(choice_4 == "1"){
            printf("the best face make-up migration is done by python\n");
            printf("it is difficult to contain it in a c++ project\n");
            printf("the effect is displayed here\n");
            Mat makeup = imread("1.png");
            imshow("make up effect display",makeup);
               }
        //simple face morphing
        if(choice_4 == "2"){
            printf("two picture morphing shows result\n");
            Morpher morpher;
            morpher.main();
            return EXIT_SUCCESS;
        }
    }
    //other effect display
    if(choice == "5"){
           printf("+---------------------------------------------------------------------+\n");
           printf("|------------------welcome to the other function display--------------|\n");
           printf("|-----------------1.face detection effect display --------------------|\n");
           printf("|--------------2.face landmark detection effct display----------------|\n");
           printf("|---------------------------------------------------------------------|\n");
           printf("+-----------------please enter your choice(1/2)-----------------------+\n");

           string choice_5;
           cin>>choice_5;
        if(choice_5 == "1"){
            printf("here is the face detection result shows\n");
             CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
            //Create an object of the Facemark class
            Ptr<Facemark> facemark = FacemarkLBF::create();
            //Load the face detector model
            facemark->loadModel("lbfmodel.yaml");
            //load the picture
            src = imread(image_directory);
            Mat frame, gray;
            frame = src.clone();// load frame
            // Container for storing rectangular frame of human face
            vector<Rect> faces;
            Rect face;
            // Convert video frames to grayscale, because the input of Face Detector is grayscale
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            // face detect
            faceDetector.detectMultiScale(gray, faces);
            // Face key point container
            int w = faces[0].width;
            int h = faces[0].height;
            Point width(0, w);
            Point height(h, 0);
            //draw the rect on picture
            Point p1 = faces[0].tl();         // 点p1
            Point p2 = faces[0].tl()+height;
            Point p3 = faces[0].tl()+width;
            Point p4 = faces[0].br();
            Mat face_show = src.clone();
            line(face_show, p1, p2, Scalar(0, 0, 255), 2);
            line(face_show, p1, p3, Scalar(0, 0, 255), 2);
            line(face_show, p2, p4, Scalar(0, 0, 255), 2);
            line(face_show, p3, p4, Scalar(0, 0, 255), 2);
            imshow("face detect",face_show);
        }
        if(choice_5 == "2"){
            printf("here is the face landmark detection result shows\n");
             CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
            //create facemark
            Ptr<Facemark> facemark = FacemarkLBF::create();
            // load face detector model
            facemark->loadModel("lbfmodel.yaml");
            src = imread(image_directory);
            Mat frame, gray;
            frame = src.clone();// Read frame
            // store face triangle
            vector<Rect> faces;
            Rect face;
            // Convert video frames to grayscale, because the input of Face Detector is grayscale
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            //face detection
            faceDetector.detectMultiScale(gray, faces);
            // contains face landmarks
            vector< vector<Point2f> > landmarks;
            vector<Point2f> landmark;
            vector<Point2f> landmar;
            // Run face keypoint detector（landmark detector）
            int i = 0;
            int j = 0;
            bool success = facemark->fit(frame,faces,landmarks);
            if(success)
            {
               //If successful, draw key points on the video frame
                for(i = 0; i < landmarks.size(); i++)
                {
                    for(j =0; j<landmarks[i].size();j++){
                            landmark.insert(landmark.end(),landmarks[i][j]);
                        }
                }
                landmar.insert(landmar.end(),landmarks[0][33]);
                landmar.insert(landmar.end(),landmarks[0][37]);
                landmar.insert(landmar.begin(),landmarks[0][44]);
                landmar.insert(landmar.begin(),landmarks[0][54]);
                landmar.insert(landmar.begin(),landmarks[0][60]);
                // Custom draw face feature point function, can draw face feature point shape / contour
                drawLandmarks(frame, landmark);
                // OpenCV comes with a key point drawing function: drawFacemarks
                drawFacemarks(frame, landmark, Scalar(0, 0, 255));
                //draw the output and the seetaface
                src1= src.clone();
                drawLandmarks(src1, landmar);
                imshow("seetaface detect",src1);
                imshow("opencv detect",frame);
                }
        }
        
    }
    waitKey(0);
    return 0;
}

