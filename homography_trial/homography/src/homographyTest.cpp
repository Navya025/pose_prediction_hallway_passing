#include "SKConfig.h"
#include "SKWrapper.h"
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <k4a/k4a.hpp>

bool keepRunning = true;

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

std::vector<cv::Point2f> generateModelPoints(int numCornersX, int numCornersY, float squareSize) {
    std::vector<cv::Point2f> modelPoints;
    for (int i = 0; i < numCornersY; ++i) {
        for (int j = 0; j < numCornersX; ++j) {
            modelPoints.emplace_back(j * squareSize, i * squareSize);
        }
    }
    return modelPoints;
}

//chatGPT created this, we need to check it against the textbooks
#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>

// Helper function to convert cv::Point2f to Eigen::Vector3d
Eigen::Vector3d toHomogeneous(const cv::Point2f& pt) {
    return Eigen::Vector3d(pt.x, pt.y, 1.0);
}

cv::Mat computeHomography(const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
    if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 4) {
        throw std::runtime_error("Insufficient or unequal number of points.");
    }

    size_t numPoints = srcPoints.size();
    Eigen::MatrixXd A(2 * numPoints, 9);
    Eigen::VectorXd b(2 * numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        Eigen::Vector3d p1 = toHomogeneous(srcPoints[i]);
        Eigen::Vector3d p2 = toHomogeneous(dstPoints[i]);

        A.row(2 * i)     << -p1(0), -p1(1), -1, 0, 0, 0, p2(0) * p1(0), p2(0) * p1(1), p2(0);
        A.row(2 * i + 1) << 0, 0, 0, -p1(0), -p1(1), -1, p2(1) * p1(0), p2(1) * p1(1), p2(1);
       
        // Right-hand side
        b(2 * i) = p2(0);
        b(2 * i + 1) = p2(1);
    }

    // Solve using least squares
    Eigen::VectorXd h = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);

    // Reshape h to a 3x3 matrix
    cv::Mat homography(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        homography.at<double>(i / 3, i % 3) = h(i);
    }

    // Ensure the homography is normalized such that h(8) is 1
    homography /= homography.at<double>(2, 2);

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
        std::vector<cv::Point2f> modelCorners = generateModelPoints(numCornersX, numCornersY, squareSize);
        cv::Mat H = computeHomography(modelCorners, corners);

        //print homography matrix
        std::cout << "Homography Matrix:" << std::endl;
        for (int i = 0; i < H.rows; i++) {
            for (int j = 0; j < H.cols; j++) {
                std::cout << H.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }

        //compare our computeHomography method to the one CV has
        cv::Mat cvH = cv::findHomography(modelCorners, corners, 0);
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