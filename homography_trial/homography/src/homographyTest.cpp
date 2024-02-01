#include "SKConfig.h"
#include "SKWrapper.h"
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <k4a/k4a.hpp>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>



std::vector<cv::Point2f> findCorners(cv::Mat grayImg, cv::Mat colorImg, cv::Size numCorners) {
std::vector<cv::Point2f> corners;
bool found = cv::findChessboardCorners(grayImg, numCorners, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
if (found) {
    // cv::drawChessboardCorners(colorImg, numCorners, cv::Mat(corners), found);
    // cv::imshow("Gray Corners", colorImg);
    // cv::waitKey(0);
} else {
    std::cerr << "Error: No chessboard found in image." << std::endl;
}
return corners;
}

//Generates a model chessboard of dimension numInternalRows by numInternalCols
//We assume the chessboard in picture is oriented normally so we generate
//points starting from the top left to the bottom right
std::vector<cv::Point2f> generateModelPoints(int numCornersX, int numCornersY) {
    std::vector<cv::Point2f> modelPoints;
    for (int i = 0; i < numCornersX; ++i) {
        for (int j = 0; j < numCornersY; ++j) {
            modelPoints.emplace_back(i, j); //Append to end of the vector (add)
        }
    }
    return modelPoints;
}

//srcPoints = modelChessboard, dstPoints = return value of findChessboardCorners
Eigen::MatrixXf constructMatrixA(const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {

    Eigen::MatrixXf A;
    A.resize(srcPoints.size() * 2, 9);

    for (int i = 0; i < srcPoints.size(); i++) {
        Eigen::VectorXf row1(9);
        Eigen::VectorXf row2(9);

        float x1 = srcPoints[i].x;
        float y1 = srcPoints[i].y;
        float x2 = dstPoints[i].x;
        float y2 = dstPoints[i].y;

        row1 << x1, y1, 1, 0, 0, 0, -1 * x1 * x2, -1 * y1 * x2, -1 * x2; //constructing rows like in notes
        row2 << 0, 0, 0, x1, y1, 1, -1 * y2 * x1, -1 * y2 * y1, -1 * y2;
        A.row(i * 2) = row1;
        A.row(i * 2 + 1) = row2;  
    }

    return A;

}

cv::Mat computeHomography(const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
    if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 4) { //need at least 4 point correspondences to calculate H
        throw std::runtime_error("Insufficient or unequal number of points.");
    }
    Eigen::MatrixXf A = constructMatrixA(srcPoints, dstPoints); //passing by reference/avoid copying vals

    //we need to solve Ah = 0, where h is the flattened homography matrix
    //We can do this by applying SVD and taking the eigenvector h, a column of V, which corresponds to smallest singular value of A

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV); //ensure Eigen has the entire V matrix available
    
    Eigen::VectorXf singularValues = svd.singularValues();

    //Assuming that the smallest singular val will be at the last index
    float smallestSingularVal = singularValues[singularValues.size() - 1];

    //Matrix V from SVD
    Eigen::MatrixXf V = svd.matrixV();

    //our solution vector h is the last column of V (corresponding to the smallest singular value)
    Eigen::VectorXf h = V.col(V.cols() - 1);
    
    //convert to cv::Mat (just for purposes of consistency)
    cv::Mat homography(3, 3, CV_32F);

    //copy data
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            homography.at<float>(i, j) = h[i * 3 + j];
        }
    }

    return homography;
}

void *captureThread(void *data) {
    SKWrapper *skw = (SKWrapper *)data;
    //while(keepRunning) {
        skw->_mostRecentPacket = SKPacket(skw);
        SKPacket packet = skw->_mostRecentPacket;
        k4a::capture cap = skw->_mostRecentPacket.getCapture();
        skw->capture(&cap);
        skw->_mostRecentPacket.setCapture(cap);
        
        k4a::image colorImg = skw->_mostRecentPacket.getColorImage();
        cv::Mat cvImg = cv::Mat(colorImg.get_height_pixels(), colorImg.get_width_pixels(), CV_8UC4, colorImg.get_buffer());
        cv::Mat grayImg;
        cv::cvtColor(cvImg, grayImg, cv::COLOR_BGR2GRAY);


        // cv::imshow("Image", grayImg);
        // cv::waitKey(1);
        cv::Size numCorners(8, 6);
        std::vector<cv::Point2f> corners = findCorners(grayImg, cvImg, numCorners);
        int numCornersX = 8; //num rows - 1
        int numCornersY = 6; //num cols - 1
        float squareSize = 10; //in centimeters
        std::vector<cv::Point2f> modelPoints = generateModelPoints(numCornersX, numCornersY);
        cv::Mat H = computeHomography(modelPoints, corners);

        //print homography matrix
        std::cout << "Homography Matrix:" << std::endl;
        for (int i = 0; i < H.rows; i++) {
            for (int j = 0; j < H.cols; j++) {
                std::cout << H.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }

        //compare our computeHomography method to the one CV has
        cv::Mat cvH = cv::findHomography(modelPoints, corners, 0);
        std::cout << "CV Homography Matrix:" << std::endl;
        for (int i = 0; i < cvH.rows; i++) {
            for (int j = 0; j < cvH.cols; j++) {
                std::cout << cvH.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }
    //}
    return NULL;
}



int main() {
    SKConfig skc;
    SKWrapper skw(skc, 0);
    // Assuming you have other necessary initializations for SKWrapper

    pthread_t threadC;
    pthread_create(&threadC, NULL, captureThread, &skw);

    // // Small delay to allow the capture thread to start and capture data
    // usleep(1000000); // 1 second, adjust as needed
    // SKPacket packet = skw.getMostRecentFrame();
    // cv::Mat image = packet.getCVMat("RGB1080p");
    // if (image.empty()) {
    //     std::cerr << "Error: Image is empty." << std::endl;
    //     keepRunning = false;
    //     pthread_join(threadC, NULL);
    //     return -1;
    // }

    // cv::imshow("Kinect Image", image);
    // cv::waitKey(0);

    // keepRunning = false;
    pthread_join(threadC, NULL);

    return 0;
}