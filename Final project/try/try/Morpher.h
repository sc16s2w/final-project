#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


class Line
{
public:
    //Start point, end point,
    Point2f P=Point2f(0.f,0.f);
    Point2f Q = Point2f(0.f, 0.f);
    Point2f M = Point2f(0.f, 0.f);
    double len{0};
    float degree{0};

    //参数a,b,p
    double a=1.0;
    double b=2.0;
    double p=1.0;

public:
    Line() :P(Point2f(0.f, 0.f)), Q(Point2f(0.f, 0.f)){};

    //Get midpoint, line length, angle information from P, Q coordinates
    void PQtoMLD();

    void MLDtoPQ();
    void show();

    double Getu(Point2f X);
    double Getv(Point2f X);
    Point2f Get_Point(double u, double v);
    double Get_Weight(Point2f X);
};


class LinePair
{
public:
    Line leftLine;
    Line rightLine;
    vector<Line> warpLine;

public:

    //Generate intermediate transition line segment collection
    void genWarpLine(int frame_count);
};


class Image
{
public:

    //Transition frame number
    int frame_index;

    Mat left_image;
    Mat right_image;
    Mat new_image;

public:
    Image(int frame_index,string leftImageName,string rightImageName);

    //Bilinear interpolation
    Vec3b bilinear(Mat image, double X, double Y);

    //Generate transition image frames
    void Warp(int frame_count, string new_image_name, vector<LinePair> pairs);
};


class Morpher
{
public:
    vector<LinePair> pairs;
    LinePair curLinePair;

    //Counting, interactive line drawing is needed
    int counter = 0;

    //Set the transition frame of the animation, for example, 1 is 50%, 3 is 25%, 50%, 75%
    int frame_count = 1;

    Mat leftImage;
    Mat rightImage;
    Mat leftImageTmp;
    Mat rightImageTmp;

    //Show related. Draw line color, line width, offset
    Scalar color = Scalar(0, 255, 0);
    int thickness = 2;
    int shift = 0;

    //Key value, used for control
    int key;

    string first_image_name;
    string second_image_name;
    string new_image_name;


public:
    void show_pairs();

    //Capture mouse actions on the left image
    static void on_mousel(int event, int x, int y, int flag, void* param);

    //Capture mouse actions on the right image
    static void on_mouser(int event, int x, int y, int flag, void* param);

    //run morph
    void runWarp();

    void main();

};
