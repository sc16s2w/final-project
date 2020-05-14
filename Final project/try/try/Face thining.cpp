//
//  Face thining.cpp
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#include <stdio.h>
#include "Face thining.h"
//Automatic thinning
/*this function is to face thinning
which based on Mls transformation */
double MAXNUM = 0x3f3f3f3;//Set infinity
Mat face_thin(Mat &src,const vector<Point2f>& landmarks,int change,const Rect &face)
{
    //to initialise the control points
    vector<Point2f> control_p = {landmarks[5],
                                 landmarks[11],
                                 Point2f(face.x,face.y+face.height),
                                 Point2f(face.x+face.width,face.y+face.height),
                                 landmarks[37],
                                 landmarks[44],
                                 landmarks[33],
                                 landmarks[62]
    };
    //to change the control points
    vector<Point2f> control_q = { Point2f(landmarks[5].x+change, landmarks[5].y),
                                Point2f(landmarks[11].x-change,landmarks[11].y),
                                Point2f(face.x,face.y+face.height),
                                 Point2f(face.x+face.width,face.y+face.height),
                                    landmarks[37],
                                landmarks[44],
                                landmarks[33],
                                landmarks[62],
                                    };
    Mat pic=src.clone();
    //to judge whether it is empty or not
    if(face.empty()){
        cout<<"empty"<<endl;
    }
    //Calculate weight wi
    for (int i =face.x; i <=face.x+face.width; ++i)
    {

        for (int j = landmarks[0].y;j< landmarks[8].y; ++j)
        {

            {
                vector<float> weight_p;                                        vector<Point2f>::iterator itcp = control_p.begin();
                while (itcp != control_p.end())
                {
                    double tmp;
                    if (itcp->x != i || itcp->y != j){
                        tmp = 1 / ((itcp->x - i)*(itcp->x - i) + (itcp->y - j)*(itcp->y - j));
                    }
                    else
                        tmp = MAXNUM;
                    weight_p.push_back(tmp);
                    itcp++;
                }
                //Calculate p * q *
                double px = 0, py = 0, qx = 0, qy = 0, tw = 0;
                itcp = control_p.begin();
                vector<float>::iterator itwp = weight_p.begin();
                vector<Point2f>::iterator itcq = control_q.begin();
                while (itcp != control_p.end())
                {
                    px += (*itwp)*(itcp->x);
                    py += (*itwp)*(itcp->y);
                    qx += (*itwp)*(itcq->x);
                    qy += (*itwp)*(itcq->y);
                    tw += *itwp;
                    ++itcp;
                    ++itcq;
                    ++itwp;
                }
                px = px / tw;
                py = py / tw;
                qx = qx / tw;
                qy = qy / tw;

                Mat A = Mat::zeros(2, 1, CV_32FC1);
                Mat B = Mat::zeros(1, 2, CV_32FC1);
                Mat C = Mat::zeros(1, 2, CV_32FC1);
                Mat sumL = Mat::zeros(2, 2, CV_32FC1);
                Mat sumR = Mat::zeros(2, 2, CV_32FC1);
                Mat M, pos;
                //Calculate Affine Matrix
                for (int i = 0; i < weight_p.size(); ++i)
                {
                    A.at<float>(0, 0) = (control_p[i].x - px);
                    A.at<float>(1, 0) = (control_p[i].y - py);
                    B.at<float>(0, 0) = weight_p[i] * ((control_p[i].x - px));
                    B.at<float>(0, 1) = weight_p[i] * ((control_p[i].y - py));
                    sumL += A * B;
                    C.at<float>(0, 0) = weight_p[i] * (control_q[i].x - qx);
                    C.at<float>(0, 1) = weight_p[i] * (control_q[i].y - qy);
                    sumR += A * C;
                }
                M = sumL.inv()*sumR;
                B.at<float>(0, 0) = i - px;
                B.at<float>(0, 1) = j - py;
                C.at<float>(0, 0) = qx;
                C.at<float>(0, 1) = qy;
                pos = B * M + C;
                int row = pos.at<float>(0, 0);
                int col = pos.at<float>(0, 1);

                pic.at<Vec3b>(col, row)[0] = src.at<Vec3b>(j, i)[0];
                pic.at<Vec3b>(col, row)[1] = src.at<Vec3b>(j, i)[1];
                pic.at<Vec3b>(col, row)[2] = src.at<Vec3b>(j, i)[2];
            }
        }
    }
    return pic;
}
