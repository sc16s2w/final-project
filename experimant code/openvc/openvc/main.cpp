#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iomanip>
using namespace std;
using namespace cv;
//filter adjust
#define CLIP_RANGE(value, min, max)  ( (value) > (max) ? (max) : (((value) < (min)) ? (min) : (value)) )
#define COLOR_RANGE(value)  CLIP_RANGE(value, 0, 255)

/**
 *  \brief Automatic brightness and contrast optimization with optional histogram clipping
 *  \param [in]src Input image GRAY or BGR or BGRA
 *  \param [out]dst Destination image
 *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
 *  \note In case of BGRA image, we won't touch the transparency
*/
void BrightnessAndContrastAuto(const Mat &src, Mat &dst, float clipHistPercent=0)
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

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

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

/**
 * Adjust Brightness and Contrast
 *
 * @param src [in] InputArray
 * @param dst [out] OutputArray
 * @param brightness [in] integer, value range [-255, 255]
 * @param contrast [in] integer, value range [-255, 255]
 *
 * @return 0 if success, else return error code
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

    /**
    Algorithm of Brightness Contrast transformation
    The formula is:
        y = [x - 127.5 * (1 - B)] * k + 127.5 * (1 + B);

        x is the input pixel value
        y is the output pixel value
        B is brightness, value range is [-1,1]
        k is used to adjust contrast
            k = tan( (45 + 44 * c) / 180 * PI );
            c is contrast, value range is [-1,1]
    */

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


void waitESC(){
    while(int key=waitKey(0) != 27){};
}

static string window_name = "photo";
static Mat src,src1,dst1;
static int brightness = 255;
static int contrast = 255;

static void callbackAdjust(int , void *)
{
    Mat dst;
    adjustBrightnessContrast(src, dst, brightness - 255, contrast - 255);
    imshow(window_name, dst);
}
//
////whitening function
void beWhite(cv::Mat &image) {
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
    int value1 = 3, value2 = 1;     //磨皮程度与细节程度的确定

    int dx = value1 * 5;    //双边滤波参数之一
    double fc = value1 * 12.5; //双边滤波参数之一
    int p = 50; //透明度
    Mat temp1, temp2, temp3, temp4;
    //双边滤波
    bilateralFilter(image, temp1, dx, fc, fc);

    temp2 = (temp1 - image + 128);

    //高斯模糊
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


//filtering effect display
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

Mat src; Mat dst;
char window_name[] = "Smoothing Demo";

//显示标题、显示图像
int display_caption(const char* caption);
int display_dst(int delay);

int main(int argc, char** argv)
{
    namedWindow(window_name, WINDOW_AUTOSIZE);
    //加载图像，默认lena.jpg
    const char* filename = "/Users/wangsiwei/Desktop/openvc/i.jpg";
    src = imread(samples::findFile(filename), IMREAD_COLOR);
    if (src.empty())
    {
        printf(" Error opening image\n");
        return EXIT_FAILURE;
    }
    //显示标题-原始图像
    if (display_caption("Original Image") != 0)
        return 0;
    //显示原图像
    dst = src.clone();
    if (display_dst(DELAY_CAPTION) != 0)
        return 0;

    //显示标题-均值滤波
    if (display_caption("Homogeneous Blur") != 0)
        return 0;
    //显示均值滤波图像，滤波器核从1到最大值，分别对原图像滤波并显示
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        blur(src, dst, Size(i, i), Point(-1, -1));
        if (display_dst(DELAY_BLUR) != 0)
            return 0;
    }

    //显示标题-高斯滤波
    if (display_caption("Gaussian Blur") != 0)
        return 0;
    //显示高斯滤波图像，滤波器核从1到最大值，分别对原图像滤波并显示
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        GaussianBlur(src, dst, Size(i, i), 0, 0);
        if (display_dst(DELAY_BLUR) != 0)
            return 0;
    }

    //显示标题-中值滤波
    if (display_caption("Median Blur") != 0)
        return 0;
    //显示中值滤波图像，滤波器核从1到最大值，分别对原图像滤波并显示
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        medianBlur(src, dst, i);
        if (display_dst(DELAY_BLUR) != 0)
            return 0;
    }

    //显示标题-双边滤波
    if (display_caption("Bilateral Blur") != 0)
        return 0;
    //显示双边滤波图像，滤波器核从1到最大值，分别对原图像滤波并显示
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        bilateralFilter(src, dst, i, i * 2, i / 2);
        if (display_dst(DELAY_BLUR) != 0)
            return 0;
    }

    //显示标题-结束
    display_caption("Done!");
    return 0;
}

//显示标题，指定显示时间
int display_caption(const char* caption)
{
    dst = Mat::zeros(src.size(), src.type());
    putText(dst, caption,
        Point(src.cols / 4, src.rows / 2),
        FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));

    return display_dst(DELAY_CAPTION);
}

//显示图像，指定显示时间
int display_dst(int delay)
{
    imshow(window_name, dst);
    int c = waitKey(delay);
    if (c >= 0) { return -1; }
    return 0;
}
//skin detection
//based onRGB
Mat RGB_detect(Mat &src)
{
    Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            int r = src.at<cv::Vec3b>(i, j)[2];
            int g = src.at<cv::Vec3b>(i, j)[1];
            int b = src.at<cv::Vec3b>(i, j)[0];

            if (r > 95 && g > 40 && b > 20 && abs(r - g) > 15 && r > g && r > b && (max(max(r, g), b) - min(min(r, g), b)) > 15)
            {
                output_mask.at<uchar>(i, j) = 255;
            }
        }
    }
    Mat detect;
    src.copyTo(detect, output_mask);;
    return detect;

}
/*基于椭圆皮肤模型的皮肤检测*/
Mat ellipse_detect(Mat& src)
{
    Mat img = src.clone();
    Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);
    //利用opencv自带的椭圆生成函数先生成一个肤色椭圆模型
    ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
    Mat ycrcb_image;
    Mat output_mask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(img, ycrcb_image, COLOR_BGR2YCrCb); //首先转换成到YCrCb空间
    for (int i = 0; i < img.cols; i++)   //利用椭圆皮肤模型进行皮肤检测
        for (int j = 0; j < img.rows; j++)
        {
            Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)   //如果该落在皮肤模型椭圆区域内，该点就是皮肤像素点
                output_mask.at<uchar>(j, i) = 255;
        }

    Mat detect;
    img.copyTo(detect,output_mask);  //返回肤色图
    return detect;
}
/*YCrCb颜色空间Cr分量+Otsu法*/
Mat YCrCb_Otsu_detect(Mat& src)
{
    Mat ycrcb_image;
    cvtColor(src, ycrcb_image, COLOR_BGR2YCrCb); //首先转换成到YCrCb空间
    Mat detect;
    vector<Mat> channels;
    split(ycrcb_image, channels);
    Mat output_mask = channels[1];
    threshold(output_mask, output_mask, 0, 255, THRESH_BINARY | THRESH_OTSU);
    src.copyTo(detect, output_mask);
    return detect;
}
/*YCrCb颜色空间Cr,Cb范围筛选法*/
Mat YCrCb_detect(Mat & src)
{
    Mat ycrcb_image;
    int Cr = 1;
    int Cb = 2;
    cvtColor(src, ycrcb_image, COLOR_BGR2YCrCb); //首先转换成到YCrCb空间
    Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
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
    Mat detect;
    src.copyTo(detect, output_mask);;
    return detect;

}
/*HSV颜色空间H范围筛选法*/
Mat HSV_detector(Mat& src)
{
    Mat hsv_image;
    int h = 0;
    int s = 1;
    int v = 2;
    cvtColor(src, hsv_image, COLOR_BGR2HSV); //首先转换成到YCrCb空间
    Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
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
    Mat detect;
    src.copyTo(detect, output_mask);;
    return detect;
}
//RGB running
//int main(){
//    Mat skin_result1;
//    Mat skin_result2;
//    Mat skin_result3;
//    Mat skin_result4;
//    Mat skin_result5;
//    Mat src = imread("/Users/wangsiwei/Desktop/毕业设计图片/美白/60.jpg");
//    if(src.empty())
//        return -1;
//   //show the original picture
//    Mat img = src.clone();
//    skin_result1 = src.clone();
//    skin_result2 = src.clone();
//    skin_result3 = src.clone();
//    skin_result4 = src.clone();
//    skin_result5 = src.clone();
//    skin_result1=RGB_detect(skin_result1);
//    skin_result2=ellipse_detect(skin_result2);
//    skin_result3=YCrCb_detect(skin_result3);
//    skin_result4=YCrCb_Otsu_detect(skin_result4);
//    skin_result5=HSV_detector(skin_result2);
//    imshow("original picture", src);
//    imshow("rgb_detect",skin_result1);
//    imshow("eclipse_detect",skin_result2);
//    imshow("yrcb_detect",skin_result3);
//    imshow("yrcb_detect_improve",skin_result4);
//    imshow("hsv_detect",skin_result5);
//    imwrite("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测图片/11_rgb.png", skin_result1);
//    imwrite("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测图片/11_eclipse.png", skin_result2);
//    imwrite("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测图片/11_Ycrcb.png", skin_result3);
//    imwrite("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测图片/11_Ycrcb_improve.png", skin_result4);
//    imwrite("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测图片/11_HSV.jpg", skin_result5);
//     waitKey(0);
//         return 0;
//}
//IOU 计算
//using namespace std;
//using namespace cv;
//
int bSums(Mat src)
{
    
    int counter = 0;
    //迭代器访问像素点
    Mat_<uchar>::iterator it = src.begin<uchar>();
    Mat_<uchar>::iterator itend = src.end<uchar>();
    for (; it!=itend; ++it)
    {
        if((*it)>0) counter+=1;//二值化后，像素点是0或者255
    }
    return counter;
}
double iou(string rgb_path, string P_path)//皮肤检测图路径， PS结果路径
{
    Mat rgb = imread(rgb_path, 0);//灰度形式读取皮肤检测,channel = 1
    Mat P = imread(P_path, 0);//灰度形式读取PS结果,channel = 1
    Mat rgb_bw, P_bw;
    compare(rgb, 10, rgb_bw, 1);//二值化，以10为阈值，这个值你可以自己调整
    compare(P, 10, P_bw, 1);//二值化，以10为阈值，这个值你可以自己调整

    Mat jiao, bing;

    bitwise_and(rgb_bw, P_bw, jiao);//获取交集
    bitwise_or(rgb_bw, P_bw, bing);//获取并集
    imshow("ff",jiao);
    imshow("dd",bing);
    double result = (countNonZero(jiao)*1.0 + 1.0) / (1.0*countNonZero(bing) + 1.0);//计算iou，分母加上1是防止除零错误
    return result;
}
//calculate IOU
//int main(){
//    double iou1 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/198_rgb.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/198.png");
//    cout<<"rgb"<<fixed << setprecision(5)<<iou1<<endl;
//    double iou3 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/198_HSV.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/198.png");
//    cout<<"hsv"<<fixed << setprecision(5)<<iou3<<endl;
//    double iou5 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/198_Ycrcb.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/198.png");
//    cout<<"ycrcb"<<fixed << setprecision(5)<<iou5<<endl;
//    double iou4 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/198_Ycrcb_improve.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/198.png");
//    cout<<"ycrcb_improve"<<fixed << setprecision(5)<<iou4<<endl;
//    double iou2 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/198_eclipse.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/198.png");
//    cout<<"eclipse"<<fixed << setprecision(5)<<iou2<<endl;
//    iou1 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/200_rgb.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/200.png");
//    cout<<"rgb"<<fixed << setprecision(5)<<iou1<<endl;
//    iou3 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/200_HSV.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/200.png");
//    cout<<"hsv"<<fixed << setprecision(5)<<iou3<<endl;
//    iou5 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/200_Ycrcb.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/200.png");
//    cout<<"ycrcb"<<fixed << setprecision(5)<<iou5<<endl;
//    iou4 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/200_Ycrcb_improve.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/200.png");
//    cout<<"ycrcb_improve"<<fixed << setprecision(5)<<iou4<<endl;
//    iou2 = iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/200_eclipse.png", "/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/200.png");
//    cout<<"eclipse"<<fixed << setprecision(5)<<iou2<<endl;
//
//}
//improved iou
//int main()
//{
//    Mat image3 = imread("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/11_rgb.png");
//    Mat image4 = imread("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/11.png");
//    imshow("original_rgb",image3);
//    imshow("original_compare",image4);
//    Mat image1 = imread("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测/11_rgb.png", 0);
//    Mat image2 = imread("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测对照图/11.png", 0);
//    Mat rgb_bw, P_bw;
//    imshow("huidu——rgb",image1);
//    imshow("huidu--compare",image2);
//    compare(image1, 10, rgb_bw, 1);//二值化，以10为阈值，这个值你可以自己调整
//    compare(image2, 10, P_bw, 1);//二值化，以10为阈值，这个值你可以自己调整
//    imshow("binary-rgb",rgb_bw);
//    imshow("binary-compare",P_bw);
//    Mat jiao, bing;
//    bitwise_and(rgb_bw, P_bw, jiao);//获取交集
//    bitwise_or(rgb_bw, P_bw, bing);//获取并集
//    imshow("jiao",jiao);
//    imshow("bing",bing);
//    double result = (countNonZero(jiao)*1.0 + 1.0) / (1.0*countNonZero(bing) + 1.0);//计算iou，分母加上1是防止除零错误
//    //double hh =iou("/Users/wangsiwei/Desktop/毕业设计图片/皮肤检测图片/11_rgb.jpg", "/Users/wangsiwei/Desktop/毕业设计图片/美白/11.jpg");
//    cout<<result<<endl;
//    waitKey(0);
//    return 0;
//
//}

Mat addSaltNoise(const Mat srcImage, int n)
{
    Mat resultImage = srcImage.clone();
    for (int k = 0; k < n; k++)
    {
        int i = rand() % resultImage.rows;
        int j = rand() % resultImage.cols;
        if (resultImage.channels() == 1)
        {
            resultImage.at<uchar>(i, j) = 0;
        }
        else
        {
            resultImage.at<Vec3b>(i, j)[0] = 0;//0为赋黑色值
            resultImage.at<Vec3b>(i, j)[1] = 0;
            resultImage.at<Vec3b>(i, j)[2] = 0;
        }
    }
    for (int k = 0; k < n; k++)
    {
        int i = rand() % resultImage.rows;
        int j = rand() % resultImage.cols;
        if (resultImage.channels() == 1)
        {
            resultImage.at<uchar>(i, j) = 255;
        }
        else
        {
            resultImage.at<Vec3b>(i, j)[0] = 255;
            resultImage.at<Vec3b>(i, j)[1] = 255;
            resultImage.at<Vec3b>(i, j)[2] = 255;
        }
    }
    return resultImage;
}
//--------------public function------------
//Mat addGuassianNoise(Mat& srcImage);//add gaussian noise
//double generateGaussianNoise(double mu, double sigma);
//
//int main()
//{
//    //------------【1】读取源图像并检查图像是否读取成功------------
//    Mat srcImage = imread("/Users/wangsiwei/Desktop/5_salt.jpg");
//    if (!srcImage.data)
//    {
//        cout << "读取图片错误，请重新输入正确路径！\n";
//        system("pause");
//        return -1;
//    }
//    imshow("【源图像】", srcImage);
//    //------------【2】给图像添加高斯噪声----------------------
//    Mat dstImage = addGuassianNoise(srcImage);
//    imshow("【高斯噪声图像】", dstImage);
//    imwrite("/Users/wangsiwei/Desktop/5_guassin.jpg", dstImage);
//    waitKey(0);
//    return 0;
//}
//
////------------add noise----------------
Mat addGuassianNoise(Mat& srcImage)
{
    Mat resultImage = srcImage.clone();
    int channels = resultImage.channels();
    int nCols = resultImage.cols*channels;
    if (resultImage.isContinuous())     {
        nCols *= nRows;
        nRows = 1;
    }
    //遍历图像中的像素
    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nCols; ++j)
        {
            //添加高斯噪声
            int val = resultImage.ptr<uchar>(i)[j] + generateGaussianNoise(2, 0.8) * 32;
            if (val < 0)
                val = 0;
            if (val > 255)
                val = 255;
            resultImage.ptr<uchar>(i)[j] = (uchar)val;
        }
    }
    return resultImage;
}

//-------------add gaussian noise---------
double generateGaussianNoise(double mu, double sigma)
{
    //定义一个特别小的值
    const double epsilon = std::numeric_limits<double>::min();//返回目标数据类型能表示的最逼近1的正数和1的差的绝对值
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假，构造高斯随机变量X
    if (!flag)
    {
        return z1*sigma + mu;
    }
    double u1, u2;
    //random filter
    do
    {
        u1 = rand()*(1.0 / RAND_MAX);
        u2 = rand()*(1.0 / RAND_MAX);
    } while (u1 <= epsilon);
 
    z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
    z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
    return z0*sigma + mu;
}
//whitening
//average calculate

//double ave(double a[],int n)
//{
//  int sum=0;
//  for (int i=0;i<n;i++)
//  sum+=a[i];
//  return sum/n;
// }
//calulate average
//int main(){
//    double rgb[] ={0.22291, 0.57808, 0.78098, 0.48089,0.83161, 0.503942, 0.174874, 0.56977, 0.18221, 0.43797, 0.82719, 0.23040, 0.39363, 0.13843, 0.54957, 0.72990, 0.39732, 0.78545, 0.34972,
//                    0.55284, 0.33319,0.53758, 0.10683, 0.60898, 0.55076, 0.57708, 0.48162, 0.45864, 0.13235, 0.17187, 0.70687, 0.54278, 0.10601, 0.50887, 0.33483, 0.53945, 0.25346, 0.71099,  0.53384,0.23006, 0.22249, 0.90783, 0.66152, 0.35780, 0.68944, 0.19282, 0.55124, 0.18907, 0.30228, 0.19414, 0.57768, 0.47075, 0.67398, 0.70193, 0.62651, 0.19670, 0.48180, 0.49115, 0.63728, 0.06439, 0.46359, 0.30596, 0.20735, 0.14211, 0.59966, 0.39073, 0.48390, 0.28391, 0.18300, 0.37061, 0.73528, 0.48435, 0.82827, 0.46715, 0.21617, 0.50180, 0.12154, 0.77140, 0.37034, 0.33282, 0.29216, 0.74561, 0.43892, 0.22548,0.84553, 0.75431, 0.48943, 0.33669, 0.70889, 0.27494, 0.32524, 0.14384, 0.00001, 0.35859, 0.00382, 0.12000, 0.63961, 0.57201, 0.49056, 0.78342, 0.25534, 0.62524, 0.63183, 0.49567, 0.81280, 0.71161, 0.76424, 0.33489, 0.09359, 0.45845, 0.51605, 0.14137, 0.70244, 0.54025,    0.80141, 0.17997, 0.14890, 0.63555, 0.50015, 0.47763, 0.38237, 0.58175, 0.83301, 0.56933, 0.61172, 0.28681, 0.48867, 0.28744, 0.44268, 0.42630, 0.67603,0.15379, 0.31879,    0.70328, 0.32643, 0.48791,  0.51279, 0.22681, 0.25508, 0.46438, 0.06756, 0.58089, 0.24091, 0.57676, 0.60854, 0.55001, 0.72521, 0.39065, 0.63489, 0.41073, 0.66821, 0.35224, 0.44487, 0.49820, 0.46764, 0.54011, 0.30988, 0.18320, 0.65278, 0.65905, 0.22131, 0.27719, 0.73916, 0.57944, 0.63493, 0.67333, 0.36022, 0.24679, 0.64812, 0.55691, 0.68201, 0.26973, 0.55702, 0.20592, 0.18940, 0.64392, 0.50452, 0.13776, 0.00003, 0.14586, 0.37685, 0.45628, 0.71211, 0.12364, 0.15878, 0.68913, 0.09416, 0.48237, 0.20796, 0.46670, 0.19610, 0.77772, 0.00001, 0.72830, 0.12787, 0.91247};
//    double HSV[]={0.32540, 0.63007, 0.76418, 0.42863, 0.63987, 0.259246, 0.15138, 0.34762, 0.20647, 0.12197, 0.82934, 0.52175, 0.25542, 0.07945, 0.36984, 0.69501, 0.32855, 0.43048, 0.23143,                   0.73749, 0.53453, 0.79632, 0.00228, 0.10856, 0.62441, 0.42495, 0.54083, 0.45352, 0.30273, 0.07769, 0.57629, 0.10528, 0.38526, 0.19775, 0.37448, 0.07128, 0.32566,                              0.53082, 0.85654, 0.04160, 0.24673, 0.87447, 0.64751, 0.34804, 0.00074, 0.17565, 0.72821, 0.17311, 0.34122, 0.11402, 0.19492, 0.45844, 0.52389, 0.45678, 0.30959,                           0.29016,  0.44728, 0.46421, 0.65436, 0.03407, 0.36484, 0.18112, 0.11444, 0.12229, 0.62579, 0.06803, 0.49443, 0.09944, 0.18347, 0.26279, 0.39677, 0.50436, 0.85944, 0.65654, 0.19808, 0.48942, 0.06839, 0.67462, 0.57405, 0.30196, 0.29622, 0.42897, 0.57548, 0.34624, 0.82345, 0.51293, 0.47741, 0.27405, 0.70942, 0.10703, 0.53344, 0.09723, 0.00002, 0.49321, 0.00918, 0.14054,  0.26383, 0.48211, 0.63899, 0.52754, 0.29116, 0.73265, 0.58496, 0.45279, 0.61800, 0.70322, 0.78831, 0.10090, 0.07406, 0.40636, 0.09110, 0.10380, 0.69587, 0.41216, 0.73078, 0.03411,    0.57230, 0.59129, 0.38127, 0.33525, 0.63914, 0.73740, 0.48690, 0.56233, 0.19218, 0.02920, 0.27239, 0.12655, 0.31136, 0.39144, 0.48494, 0.68552, 0.42355, 0.36764, 0.71985, 0.26492, 0.15190, 0.43432, 0.00163, 0.28239, 0.30787, 0.34001, 0.58273, 0.64218, 0.73296, 0.49570, 0.59903, 0.41696, 0.36115, 0.20455, 0.22683, 0.50097, 0.61338, 0.35406, 0.59375, 0.14219, 0.81240, 0.66310,  0.43087, 0.29309, 0.31399, 0.54893, 0.67980, 0.57743, 0.63867, 0.26417, 0.53510, 0.58421, 0.15927, 0.23178, 0.57825, 0.02626, 0.16933, 0.65806, 0.00568, 0.00003, 0.16913, 0.12979, 0.36779,  0.74344, 0.16225, 0.22495, 0.64614, 0.01401, 0.23310, 0.33291, 0.68455, 0.00020, 0.80280, 0.00001, 0.75119, 0.00945, 0.86881 };
//    double ycrcb[]={0.17446, 0.81550, 0.20109, 0.35094, 0.67479, 0.258112, 0.243416, 0.29086, 0.09007,    0.22477, 0.84243, 0.35683, 0.45933, 0.13160, 0.39639, 0.50330, 0.32812, 0.75770, 0.17088, 0.67052, 0.53453, 0.79632, 0.00228, 0.10856, 0.62441, 0.42495, 0.54083, 0.45352, 0.30273, 0.05502, 0.50119,    0.25592, 0.14629, 0.73178, 0.32160, 0.84241, 0.20341, 0.39894, 0.71346, 0.06058, 0.19211, 0.69592,    0.77695, 0.30741, 0.09972, 0.15481, 0.40752, 0.13519, 0.50625, 0.15667, 0.45219, 0.26823, 0.42143,    0.44538, 0.49062, 0.31125, 0.22322, 0.24720, 0.55897, 0.07610, 0.50253, 0.23955, 0.11457, 0.13488,    0.41725, 0.16416, 0.35913, 0.22104, 0.08923, 0.23810, 0.85177, 0.34711, 0.61373, 0.39268, 0.39131,        0.29316, 0.11657, 0.71997, 0.29314, 0.19508, 0.23379, 0.33097, 0.55794, 0.18430, 0.86354, 0.55245,    0.27432, 0.18293, 0.66399, 0.29199, 0.20530, 0.23239, 0.11352, 0.33220, 0.48989, 0.12251, 0.68268,    0.52032, 0.57703, 0.29503, 0.17093, 0.27187, 0.39770, 0.44619, 0.66502, 0.54778, 0.70716, 0.17218,    0.07746, 0.18643, 0.25607, 0.10538, 0.63459, 0.53610, 0.67057, 0.33577, 0.15740, 0.41732, 0.38123,    0.33887, 0.37631, 0.82777, 0.24504, 0.18206, 0.61379, 0.20970, 0.11809, 0.12171, 0.44514, 0.32145,    0.57608, 0.10449, 0.38932, 0.53438, 0.28329, 0.48543, 0.69904, 0.33353, 0.17357, 0.30635, 0.12219,    0.67410, 0.16464, 0.57985, 0.55009, 0.42775, 0.60193, 0.33641, 0.53700, 0.25956, 0.57714, 0.23971,    0.46554, 0.30949, 0.47819, 0.25269, 0.40431, 0.18823, 0.61046, 0.64618, 0.86506, 0.17929, 0.77091,    0.21119, 0.38101, 0.69043, 0.26486, 0.23932, 0.40682, 0.42006, 0.62402, 0.54221, 0.47785, 0.04844,    0.21725, 0.54823, 0.44658, 0.06496, 0.00003, 0.11675, 0.23953, 0.33087, 0.55795, 0.16073, 0.13774, 0.54077, 0.08565, 0.31162, 0.51772, 0.66013, 0.18045, 0.73724, 0.00001, 0.66164, 0.04611, 0.89770};
//    double ycrcb_improve[]={0.41680, 0.77223, 0.82258, 0.48959, 0.82915, 0.717119, 0.0223206, 0.50066, 0.14217, 0.62760, 0.85815, 0.18308, 0.37047, 0.23220, 0.42213, 0.73990, 0.30581, 0.78265, 0.23964, 0.53421, 0.27269, 0.52423, 0.08309, 0.52564, 0.59837, 0.53923, 0.44907, 0.42709, 0.12413, 0.44141, 0.62636, 0.37477, 0.43192, 0.38346, 0.27581, 0.79492, 0.19302, 0.72592, 0.00289, 0.27035, 0.20757, 0.90434, 0.70126, 0.30893, 0.22933, 0.20596, 0.82486, 0.22630, 0.24440, 0.84483, 0.42667, 0.44202, 0.57890, 0.46074, 0.33672, 0.14329, 0.55053, 0.47216, 0.63770, 0.14670,    0.00704, 0.24768, 0.34673, 0.15072, 0.41361, 0.63730, 0.50077, 0.29835, 0.53455, 0.28407, 0.86248, 0.17479, 0.83057, 0.24325, 0.02018, 0.56783, 0.11288, 0.76367, 0.30142, 0.64825, 0.26129, 0.19940, 0.44987, 0.00900, 0.81151, 0.75152, 0.11667, 0.45186, 0.63695, 0.17805, 0.19210, 0.14844, 0.11534, 0.03521, 0.39802, 0.18340, 0.68496, 0.09477, 0.66508, 0.25978, 0.27579, 0.20996,   0.41441, 0.02361, 0.78069, 0.71976, 0.74863, 0.67946, 0.27451, 0.15265, 0.25678, 0.14441, 0.72030, 0.65539, 0.24560, 0.31348, 0.12475, 0.65536, 0.51687, 0.07614, 0.32589, 0.17091, 0.52907,    0.80487, 0.61521, 0.26347, 0.70147, 0.65739, 0.47000, 0.33087, 0.21353, 0.32212, 0.02051, 0.01847, 0.30130, 0.48096, 0.41421, 0.15904, 0.31552, 0.60981, 0.06231, 0.63801, 0.02240, 0.53996, 0.61853, 0.69513, 0.74014, 0.42982, 0.56351, 0.63500, 0.70439, 0.36584, 0.25509, 0.53496, 0.01064, 0.48894, 0.05326, 0.15085, 0.66757, 0.59106, 0.71052, 0.27652, 0.18688, 0.57780, 0.75655, 0.64158, 0.29476, 0.42296, 0.36003, 0.64445, 0.69260, 0.00439, 0.37047, 0.31246, 0.25770, 0.23247, 0.26982, 0.24331, 0.09163, 0.18112, 0.14756, 0.46465, 0.72888, 0.10912, 0.20592, 0.79285, 0.00902, 0.36368, 0.01226, 0.65907, 0.18045, 0.80113, 0.25560, 0.77427, 0.00199, 0.90809};
//    double eclipse[]={0.28921, 0.64080, 0.76343, 0.76343, 0.72567, 0.217874, 0.174776, 0.374858, 0.195255, 0.10022, 0.844798, 0.289478, 0.48950, 0.07355, 0.39612, 0.69377, 0.25167, 0.42291,    0.20651, 0.42291, 0.31055, 0.76231, 0.03820, 0.11228, 0.57450, 0.41585, 0.55497, 0.43844, 0.26629, 0.07740, 0.63516, 0.28406, 0.29137, 0.38009, 0.38009, 0.24656, 0.31146, 0.52087, 0.83636, 0.03976, 0.21974, 0.88560, 0.69573, 0.35073, 0.29013, 0.17994, 0.64091, 0.14998, 0.35969, 0.06340, 0.39460, 0.44657, 0.53092, 0.51821, 0.38423, 0.35019, 0.42855, 0.37191, 0.65294, 0.03621, 0.47395, 0.23890, 0.08592, 0.11795, 0.61680, 0.06192, 0.49288, 0.09548, 0.05185, 0.27368, 0.46997, 0.47809, 0.85606, 0.61758, 0.04853, 0.46469, 0.06775, 0.68585, 0.50251, 0.30057, 0.29326, 0.53512, 0.58040, 0.15989, 0.84885, 0.52331, 0.46342, 0.24762, 0.69723, 0.13557, 0.53056, 0.09658, 0.00002, 0.38387, 0.00903, 0.14274, 0.27109, 0.54903, 0.63957, 0.52334, 0.28804, 0.73265, 0.65436, 0.48615, 0.61319, 0.74981, 0.77665, 0.10135, 0.06045, 0.53644, 0.09992, 0.10514, 0.69471, 0.53938, 0.73091, 0.25692, 0.10886, 0.65932, 0.60396, 0.42812, 0.37268, 0.70226, 0.56933, 0.43318, 0.56549, 0.06413, 0.19371, 0.02762, 0.45131, 0.26307, 0.51436, 0.21257, 0.48224, 0.63548, 0.36334, 0.37200, 0.72457, 0.29256, 0.14970, 0.43289, 0.00204, 0.39185, 0.16326, 0.32435, 0.59873, 0.63381, 0.72480, 0.52364, 0.59707, 0.36968, 0.66866, 0.19185, 0.31212, 0.49053, 0.55660, 0.46274, 0.69184, 0.16735, 0.80531, 0.65669, 0.53082, 0.29274, 0.31687, 0.54858, 0.64721, 0.52929, 0.44306, 0.23249, 0.50684, 0.58656, 0.28478, 0.23145, 0.55601, 0.02582, 0.15549, 0.62132, 0.44834, 0.00466, 0.00003, 0.16737, 0.16426, 0.48275, 0.73884, 0.26299, 0.20225, 0.72046, 0.01306, 0.32251, 0.41143, 0.65712, 0.00094, 0.80082, 0.00001, 0.72313, 0.01513, 0.87559};
//
//    //int i = 197;
//    //double ave_rgb = ave(rgb, i);
//    double sum = 0;
//    for(int i=0; i<196;i++){
//        sum+=rgb[i];
//    }
//    double ave_rgb=sum/197;
//    cout<<ave_rgb<<endl;
//}
