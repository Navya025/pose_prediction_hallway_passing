#include "SKConfig.h"
#include "SKWrapper.h"
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <k4a/k4a.hpp>
#include <Eigen/Dense>
#include <vector>

bool keepRunning = true;

std::vector<cv::Point2f> findCorners(cv::Mat grayImg, cv::Mat colorImg, cv::Size numCorners) {
std::vector<cv::Point2f> corners;
bool found = cv::findChessboardCorners(grayImg, numCorners, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
if (!found) {
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

// A5.1:

// Function to find the least-squares solution for the over-determined system
VectorXd leastSquaresSolution(Eigen::MatrixXd A, Eigen::VectorXd b) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd D = svd.singularValues().asDiagonal();
    Eigen::MatrixXd Vt = svd.matrixV().transpose();

    // Find SVD A = U * D * V^T
    Eigen::MatrixXd reconstructedA = U * D * Vt;

    // Set b' = U^T * b
    Eigen::VectorXd b_prime = U.transpose() * b;
    
    // Find the vector y: yi = b'i/di
    Eigen::VectorXd y(D.rows());
    for (int i = 0; i < D.rows(); ++i) {
        y(i) = b_prime(i) / D(i, i);
    }

    // Computing the solution x = Vy
    Eigen::VectorXd x = Vt.transpose() * y;

    return x;
}

// Function to find the general solution for a set of equations with unknown rank
VectorXd generalSolution(const MatrixXd& A, const VectorXd& b) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd D = svd.singularValues().asDiagonal();
    Eigen::MatrixXd Vt = svd.matrixV().transpose();

    // Find SVD A = U * D * V^T
    Eigen::MatrixXd reconstructedA = U * D * Vt;

    // Set b' = U^T * b
    Eigen::VectorXd b_prime = U.transpose() * b;

    // Find the vector y: yi = b'i/di for i = 1,...,r, and yi = 0 otherwise
    int rank = svd.rank();
    Eigen::VectorXd y(rank);
    for (int i = 0; i < rank; ++i) {
        y(i) = (D(i, i) != 0.0) ? b_prime(i) / D(i, i) : 0.0;
    }

    // The solution x of minimum norm ||x|| is Vy
    Eigen::VectorXd x_min_norm = Vt.transpose() * y;

    // λr+1,..., λn terms
    Eigen::VectorXd lambda = svd.singularValues().tail(A.cols() - rank);

    // Additional columns of V beyond rank
    Eigen::MatrixXd V_tail = Vt.transpose().rightCols(A.cols() - rank);

    // General solution: x = Vy + λr+1vr+1 + ... + λnvn
    Eigen::VectorXd general_solution = x_min_norm;
    for (int k = 0; k < lambda.size(); ++k) {
        general_solution += lambda(k) * V_tail.col(k);
    }

    return general_solution;
}


//chatGPT created this, we need to check it against the textbooks
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
    skw->_mostRecentPacket = SKPacket(skw);
    SKPacket packet = skw->_mostRecentPacket;
    k4a::capture cap = skw->_mostRecentPacket.getCapture();
    skw->capture(&cap);
    skw->_mostRecentPacket.setCapture(cap);
        
    k4a::image colorImg = skw->_mostRecentPacket.getColorImage();
    cv::Mat cvImg = cv::Mat(colorImg.get_height_pixels(), colorImg.get_width_pixels(), CV_8UC4, colorImg.get_buffer());
    cv::Mat homoraphy = getHomography(cvImage);
    
    printHomography(homography);
    return NULL;
}

cv::Mat getHomography(cv::Mat cvImage){
    cv::Mat grayImg;
    cv::cvtColor(cvImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::Size numCorners(8, 6);
    std::vector<cv::Point2f> corners = findCorners(grayImage, cvImage, numCorners);
    int numCornersX = 8; //num rows - 1
    int numCornersY = 6; //num cols - 1
    float squareSize = 10; //in centimeters
    std::vector<cv::Point2f> modelCorners = generateModelPoints(numCornersX, numCornersY, squareSize);
    compareHomography(modelCorners, corners);
    return computeHomography(modelCorners, corners);
}

//print homography matrix
void printHomography(cv::Mat Homography) {
    std::cout << "Homography Matrix:" << std::endl;
    for (int i = 0; i < Homography.rows; i++) {
        for (int j = 0; j < Homography.cols; j++) {
            std::cout << Homography.at<double>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

//compare our computeHomography method to the one CV has
void compareHomography(std::vector<cv::Point2f> modelCorners, std::vector<cv::Point2f> corners) {
            cv::Mat cvH = cv::findHomography(modelCorners, corners, 0);
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

    pthread_t threadC;
    pthread_create(&threadC, NULL, captureThread, &skw);
    pthread_join(threadC, NULL);

    return 0;
}