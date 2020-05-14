//
//  Face detect.cpp
//  try
//
//  Created by 王思为 on 2020/5/13.
//  Copyright © 2020 王思为. All rights reserved.
//

#include <stdio.h>
#include "Face detect.h"
/*this function is to show face detection result, which can daw polyline,
 Draw polygon lines by connecting consecutive points between
 the start and end indexes.*/
void drawPolyline
(
  Mat &img,
  const vector<Point2f> &landmarks,
  const int start,
  const int end,
  bool isClosed = false
)
{
    // Collect all points between the start and end index
    vector <Point> points;
    for (int i = start; i <= end; i++)
    {
        points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
    }

    //Draw polygon curve
    polylines(img, points, isClosed, COLOR, 2, 16);

}
//Draw key points of face
void drawLandmarks(Mat &im, vector<Point2f> &landmarks)
{
    // Draw 68 points and outlines on the face (the order of points is specific and has attributes
    if (landmarks.size() == 68)
    {
      // Jaw line
      drawPolyline(im, landmarks, 0, 16);
      // Left eyebrow
      drawPolyline(im, landmarks, 17, 21);
      // Right eyebrow
      drawPolyline(im, landmarks, 22, 26);
      // Nose bridge
      drawPolyline(im, landmarks, 27, 30);
      // Lower nose
      drawPolyline(im, landmarks, 30, 35, true);
      // Left eye
      drawPolyline(im, landmarks, 36, 41, true);
      // Right Eye
      drawPolyline(im, landmarks, 42, 47, true);
      // Outer lip
      drawPolyline(im, landmarks, 48, 59, true);
      // Inner lip
      drawPolyline(im, landmarks, 60, 67, true);
    }
    else
    {
        // If the key points of the face are not 68, we do not know which points correspond to which facial features. So, we draw a circle for each landamrk.
        for(int i = 0; i < landmarks.size(); i++)
        {
            circle(im,landmarks[i],3, COLOR, FILLED);
        }
    }

}
/* judge function, to see whether it has a change or not*/
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims||
        mat1.channels()!=mat2.channels()) {
        return false;
    }
    if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
        return false;
    }
    int nrOfElements1 = mat1.total()*mat1.elemSize();
    if (nrOfElements1 != mat2.total()*mat2.elemSize()) return false;
    bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
    return lvRet;
}
