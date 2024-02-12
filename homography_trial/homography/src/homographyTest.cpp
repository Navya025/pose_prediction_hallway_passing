#include "SKConfig.h"
#include "SKWrapper.h"
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <k4a/k4a.hpp>
#include <Eigen/Dense>
#include <vector>



//Generates a model chessboard of dimension numInternalRows by numInternalCols
//We assume the chessboard in picture is oriented normally so we generate
//points starting from the top left to the bottom right
std::vector<cv::Point2d> generateModelPoints(int numCornersX, int numCornersY) {
    std::vector<cv::Point2d> modelPoints;
    for (int i = 0; i < numCornersX; ++i) {
        for (int j = 0; j < numCornersY; ++j) {
            modelPoints.emplace_back(i, j); //Append to end of the vector (add)
        }
    }
    return modelPoints;
}

//srcPoints = modelChessboard, dstPoints = return value of findChessboardCorners
Eigen::MatrixXd constructMatrixA(const std::vector<cv::Point2d>& srcPoints, const std::vector<cv::Point2d>& dstPoints) {

    Eigen::MatrixXd A;
    A.resize(srcPoints.size() * 2, 9);

    for (int i = 0; i < srcPoints.size(); i++) {
        Eigen::VectorXd row1(9);
        Eigen::VectorXd row2(9);

        double x1 = srcPoints[i].x;
        double y1 = srcPoints[i].y;
        double x2 = dstPoints[i].x;
        double y2 = dstPoints[i].y;

        row1 << x1, y1, 1, 0, 0, 0, -1 * x1 * x2, -1 * y1 * x2, -1 * x2; //constructing rows according to algorithm
        row2 << 0, 0, 0, x1, y1, 1, -1 * y2 * x1, -1 * y2 * y1, -1 * y2;
        A.row(i * 2) = row1;
        A.row(i * 2 + 1) = row2;  
    }

    return A;

}

void printMatrix(Eigen::MatrixXd A) {
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
     }
}

cv::Mat computeHomography(const std::vector<cv::Point2d>& srcPoints, const std::vector<cv::Point2d>& dstPoints) {
    if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 4) { //need at least 4 point correspondences to calculate H
        throw std::runtime_error("Insufficient or unequal number of points.");
    }
    Eigen::MatrixXd A = constructMatrixA(srcPoints, dstPoints); //passing by reference/avoid copying vals
    //we need to solve Ah = 0, where h is the flattened homography matrix
    //We can do this by applying SVD and taking the eigenvector h, a column of V, which corresponds to smallest singular value of A

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV); //ensure Eigen has the entire V matrix available
    

    Eigen::VectorXd singularValues = svd.singularValues();
    

    //Assuming that the smallest singular val will be at the last index
    double smallestSingularVal = singularValues[singularValues.size() - 1];

    //Matrix V from SVD
    Eigen::MatrixXd V = svd.matrixV();

    //our solution vector h is the last column of V (corresponding to the smallest singular value)
    Eigen::VectorXd h = V.col(V.cols() - 1);
    cv::Mat_<double> homography(3, 3, CV_64F);
    //copy data
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            homography.at<double>(i, j) = h[i * 3 + j];
        }
    }

    //normalize data and round to nearest thousandth
    double h33 = homography.at<double>(2, 2);
    for (int i = 0; i < homography.rows; ++i) {
        for (int j = 0; j < homography.cols; ++j) {
            double& val = homography.at<double>(i, j);
            val /= h33;
            val = std::round(val * 1000) / 1000; //rounding
            if (val == 0.0d) {
                val = 0.0d; //making zeroes print as positive zero due to FP arithmetic
            }
        }
    }

    return homography;
}

void compareHomographies(cv::Mat cvH, cv::Mat H) {
        //print homography matrix
        std::cout << "Homography Matrix:" << std::endl;
        for (int i = 0; i < H.rows; i++) {
            for (int j = 0; j < H.cols; j++) {
                std::cout << H.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }

        //compare our computeHomography method to the one CV has
        std::cout << "CV Homography Matrix:" << std::endl;
        for (int i = 0; i < cvH.rows; i++) {
            for (int j = 0; j < cvH.cols; j++) {
                std::cout << cvH.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }
}

// Function to apply the homography transformation and test correctness
void testHomography(const std::vector<cv::Point2d>& sourcePoints,
                    const std::vector<cv::Point2d>& destPoints,
                    const cv::Mat& homographyMatrix) {
    // Check if sourcePoints and destPoints have the same size
    if (sourcePoints.size() != destPoints.size()) {
        std::cerr << "Error: sourcePoints and destPoints must have the same size." << std::endl;
        return;
    }

    // Iterate through each point and apply the homography transformation
    for (size_t i = 0; i < sourcePoints.size(); ++i) {
        // Convert source point to homogeneous coordinates
        cv::Mat srcHomogeneous = (cv::Mat_<double>(3, 1) << sourcePoints[i].x, sourcePoints[i].y, 1.0);
       
        // Apply the homography transformation
        cv::Mat dstHomogeneous = homographyMatrix * srcHomogeneous;
       
        // Convert back to Cartesian coordinates
        double w = dstHomogeneous.at<double>(2, 0);
        cv::Point2d transformedPoint(dstHomogeneous.at<double>(0, 0) / w,
                                     dstHomogeneous.at<double>(1, 0) / w);
       
        // Compare the transformed point to the destination point
        const double epsilon = 100; // Tolerance for comparing equality (in pixels)
        if (std::abs(transformedPoint.x - destPoints[i].x) <= epsilon &&
            std::abs(transformedPoint.y - destPoints[i].y) <= epsilon) {
            std::cout << "Point " << i << " matches: " << transformedPoint << " ~ " << destPoints[i] << std::endl;
        } else {
            std::cout << "Point " << i << " does not match: " << transformedPoint << " != " << destPoints[i] << std::endl;
        }
    }
}

void drawPoints(cv::Mat& image, std::vector<cv::Point2d>& points, const cv::Scalar& color) {
    for (const auto& point : points) {
        cv::circle(image, point, 5, color, -1); // Draw circle for each point
    }
}

void showHomography(std::vector<cv::Point2d>& modelPoints, std::vector<cv::Point2d>& points, const cv::Mat& homography, cv::Mat& image) {
    // If the image is empty, use a black background
    if (image.empty()) {
        image = cv::Mat::zeros(1080, 1920, CV_8UC3);
    }

    // Draw original destination points on the image
    drawPoints(image, points, cv::Scalar(0, 255, 0)); // Draw in green for original points

    // Convert points to float for perspectiveTransform
    std::vector<cv::Point2f> pointsFloat(modelPoints.begin(), modelPoints.end());
    std::vector<cv::Point2f> transformedPoints; // To store transformed points

    // Apply the homography to the dstPoints
    cv::perspectiveTransform(pointsFloat, transformedPoints, homography);

    //convert back to doubles to pass to drawPoints
    std::vector<cv::Point2d> transformedPointsDouble;
    for (const auto& pt : transformedPoints) {
    transformedPointsDouble.emplace_back(static_cast<double>(pt.x), static_cast<double>(pt.y));
    }


    // Draw transformed points on a new image
    cv::Mat black = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);

    drawPoints(black, transformedPointsDouble, cv::Scalar(0, 0, 255)); // Draw in red for transformed points
    
    //Display the image
    cv::imshow("Original Points", image);
    cv::imshow("Transformed Points", black);
    cv::waitKey(0); // Wait for a key press
}


void *captureThread(void *data) {
    //SKWrapper *skw = (SKWrapper *)data;
    //while(keepRunning) {
        // skw->_mostRecentPacket = SKPacket(skw);
        // SKPacket packet = skw->_mostRecentPacket;
        // k4a::capture cap = skw->_mostRecentPacket.getCapture();
        // skw->capture(&cap);
        // skw->_mostRecentPacket.setCapture(cap);
        
        // k4a::image colorImg = skw->_mostRecentPacket.getColorImage();
        // cv::Mat cvImg = cv::Mat(colorImg.get_height_pixels(), colorImg.get_width_pixels(), CV_8UC4, colorImg.get_buffer());
        // cv::Mat grayImg;
        // cv::cvtColor(cvImg, grayImg, cv::COLOR_BGR2GRAY);


        // cv::imshow("Image", grayImg);
        // cv::waitKey(1);
        cv::Size numCorners(8, 6);
        int numCornersX = 8;
        int numCornersY = 6;
        // std::vector<cv::Point2d> corners = findCorners(grayImg, cvImg, numCorners);
        // std::vector<cv::Point2d> modelPoints = generateModelPoints(numCornersX, numCornersY);

        std::vector<cv::Point2d> corners = {
        cv::Point2d(625.0d, 625.0d),
        cv::Point2d(875.0d, 625.0d),
        cv::Point2d(1000.0d, 875.0d),
        cv::Point2d(625.0d, 875.0d)
    };

        std::vector<cv::Point2d> modelPoints = {
        cv::Point2d(500.0d, 500.0d),
        cv::Point2d(750.0d, 500.0d),
        cv::Point2d(750.0d, 750.0d),
        cv::Point2d(500.0d, 750.0d)
    };

        cv::Mat H = computeHomography(modelPoints, corners);
        cv::Mat cvH = cv::findHomography(modelPoints, corners, 0);
        cv::Mat dud;
        // std::cout << "CV's homography" << std::endl;
        // std::cout << std::endl;
        // testHomography(modelPoints, corners, cvH);
        // std::cout << std::endl;
        // std::cout << std::endl;
        // std::cout << "Our homography" << std::endl;
        // testHomography(modelPoints, corners, H);
        showHomography(modelPoints, corners, H, dud);

    return NULL;
}

void groundTruthHomography(){
    std::vector<cv::Point2d> srcPoints = {
        cv::Point2d(0.0d, 0.0d),
        cv::Point2d(1.0d, 0.0d),
        cv::Point2d(1.0d, 1.0d),
        cv::Point2d(0.0d, 1.0d)
    };
    std::vector<cv::Point2d> dstPoints = {
        cv::Point2d(0.5d, 0.5d),
        cv::Point2d(1.5d, 0.5d),
        cv::Point2d(1.5d, 1.5d),
        cv::Point2d(0.5d, 1.5d)
    };
    cv::Mat H = computeHomography(srcPoints, dstPoints);
    cv::Mat cvH = findHomography(srcPoints, dstPoints, 0);

    std::cout << "Homography Matrix:" << std::endl;
        for (int i = 0; i < H.rows; i++) {
            for (int j = 0; j < H.cols; j++) {
                std::cout << H.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }

    std::cout << "CV Homography Matrix:" << std::endl;
        for (int i = 0; i < cvH.rows; i++) {
            for (int j = 0; j < cvH.cols; j++) {
                std::cout << cvH.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }
}

int main() {
    SKConfig skc;
    SKWrapper skw(skc, 0);
    // Assuming you have other necessary initializations for SKWrapper

    pthread_t threadC;
    pthread_create(&threadC, NULL, captureThread, &skw);
    pthread_join(threadC, NULL);
    
    return 0;
}