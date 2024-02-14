#include "SKConfig.h"
#include "SKWrapper.h"

#include <k4a/k4a.hpp>

#include <opencv2/opencv.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>  

#include <ceres/ceres.h>        
#include <ceres/rotation.h>  
#include <glog/logging.h>  

#include <random>
#include <vector>
#include <cmath>

#include <unistd.h>

using namespace std;

Eigen::MatrixXd cvToEigen(vector<cv::Point2d> cvVec) {
Eigen::MatrixXd matrix (3, 48);

for (size_t i = 0; i < cvVec.size(); i++) {
    matrix(0, i) = cvVec[i].x;
    matrix(1, i) = cvVec[i].y;
    matrix(2, i) = 1.0;
}
return matrix;
}

vector<cv::Point2d> eigenToCv(Eigen::MatrixXd mat) {
    vector<cv::Point2d> result;
     for (int i = 0; i < mat.cols(); ++i) {
            cv::Point2d curr;
            curr = cv::Point2d(mat(0, i), mat(1, i));
            result.emplace_back(curr);
        }
    return result;
}

//Generates a model chessboard of dimension numInternalRows by numInternalCols
//We assume the chessboard in picture is oriented normally so we generate
//points starting from the top left to the bottom right
Eigen::MatrixXd generateModelChessboard(int rows, int cols, double width) {
    double xOffset = (double)rows * width;
    double yOffset = (double)cols * width;
    xOffset = -xOffset / 2.0;
    yOffset = -yOffset / 2.0;
    Eigen::MatrixXd modelCB(3,rows*cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
                modelCB(0, i * cols + j) = xOffset + (double) j * width;
                modelCB(1, i * cols + j) = yOffset + (double) i * width;
                modelCB(2, i * cols + j) = 1.0;
        }
    }

    return modelCB;
}

void roundNums(Eigen::MatrixXd &mat) {
    
    mat = mat.unaryExpr([](auto elem) {
        return round(elem * 10000.0) / 10000.0;
    });
}

void printMatrix(Eigen::MatrixXd A) {
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
     }
}




std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generateRandomTransform(double xW, double yW, double zMin, double zMax) {
    std::random_device rd; 
    std::mt19937 gen(rd()); 

    Eigen::MatrixXd rotation(4,1);
    Eigen::MatrixXd translation(3,1);

    for(int i = 0; i < 4; i++) {
        std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range
        rotation(i,0) = distr(gen); // Define the range
    }

    std::uniform_real_distribution<> distX(-xW, xW); // Define the range
    std::uniform_real_distribution<> distY(-yW, yW); // Define the range
    std::uniform_real_distribution<> distZ(zMin, zMax); // Define the range
    translation(0,0) = distX(gen);
    translation(1,0) = distY(gen);
    translation(2,0) = distZ(gen);

    rotation.normalize();
    return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>(rotation, translation);
}

//could templatize this function, but need to templatize all matrices
Eigen::MatrixXd computeHomographyFromTransform(std::pair<Eigen::MatrixXd, Eigen::MatrixXd> tf) {
    Eigen::MatrixXd rotVect = tf.first;
    
    Eigen::Quaterniond q(rotVect(0,0), rotVect(1,0), rotVect(2,0), rotVect(3,0)); //make sure order is consistent w lib
    Eigen::Matrix3d rotationMatrix = q.normalized().toRotationMatrix();

    Eigen::MatrixXd H(3,3);
    H.block(0,0, 3,1) = rotationMatrix.block(0,0,3,1);
    H.block(0,1, 3,1) = rotationMatrix.block(0,1,3,1);
    H.block(0,2, 3,1) = tf.second;

    //use quaternion, normalize the vector to 1, don't normalize

    return H;
}

Eigen::MatrixXd constructMatrixA(Eigen::MatrixXd srcPts, Eigen::MatrixXd dsPts) {
    Eigen::MatrixXd A(srcPts.cols() * 2, 9);

    for (int i = 0; i < srcPts.cols(); i++) {
        double x1 = srcPts(0,i);
        double y1 = srcPts(1,i);
        double x2 = dsPts(0,i);
        double y2 = dsPts(1,i);

        A.row(i * 2) << x1, y1, 1, 0, 0, 0, -1 * x1 * x2, -1 * y1 * x2, -1 * x2; //constructing rows according to algorithm
        A.row(i * 2 + 1) << 0, 0, 0, x1, y1, 1, -1 * y2 * x1, -1 * y2 * y1, -1 * y2;
    }

    return A;

}

Eigen::MatrixXd computeHomography(Eigen::MatrixXd srcPts, Eigen::MatrixXd dsPts) {
    Eigen::MatrixXd A = constructMatrixA(srcPts, dsPts);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd singularValues = svd.singularValues();
    double smallestSingularVal = singularValues[singularValues.size() - 1];
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd h = V.col(V.cols() - 1);
    Eigen::MatrixXd outH(3,3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
           outH(i, j) = h(i * 3 + j);
        }
    }

    outH /= outH(2,2);

    return outH;
}

vector<pair<Eigen::Vector2d, Eigen::Vector2d>> makeCorrespondences(Eigen::MatrixXd src, Eigen::MatrixXd dst) {
    //ASSUMES SRC AND DST ARE OF THE SAME SIZE (2 ROWS By N COLUMNS)
    //each column of each matrix represents X, Y coordinates (row 0 = x, row 1 = y)
    vector<pair<Eigen::Vector2d, Eigen::Vector2d>> result;
    result.reserve(src.cols()); //allocate space

    for (int i = 0; i < src.cols(); i++) {
        Eigen::Vector2d currSrc = src.col(i);
        Eigen::Vector2d currDst = dst.col(i);
        result.emplace_back(make_pair(currSrc, currDst));
    }

    return result;
}

Eigen::Matrix3f getIntrinsic(const k4a::calibration &cal) {
    const k4a_calibration_intrinsic_parameters_t::_param &i = cal.color_camera_calibration.intrinsics.parameters.param;
    Eigen::Matrix3f camera_matrix = Eigen::Matrix3f::Identity();
    camera_matrix(0, 0) = i.fx;
    camera_matrix(1, 1) = i.fy;
    camera_matrix(0, 2) = i.cx;
    camera_matrix(1, 2) = i.cy;
    return camera_matrix;
}

Eigen::MatrixXd homToCart(Eigen::MatrixXd srcPts) {
    Eigen::MatrixXd dstPts(2, srcPts.cols());
    dstPts.row(0) = srcPts.row(0).array() / srcPts.row(2).array();
    dstPts.row(1) = srcPts.row(1).array() / srcPts.row(2).array();
    return dstPts;
}

struct HomographyReprojectionError {
    HomographyReprojectionError(double observed_x, double observed_y, double projected_x, double projected_y)
        : observed_x(observed_x), observed_y(observed_y), projected_x(projected_x), projected_y(projected_y) {}

    template <typename T>
    bool operator()(const T* const homography, T* residuals) const {
        // Reconstruct the homography matrix H from the parameter vector
        Eigen::Matrix<T, 3, 3> H;

        H << homography[0], homography[1], homography[2],
             homography[3], homography[4], homography[5],
             homography[6], homography[7], T(1);

        // Apply homography to the observed point
        Eigen::Matrix<T, 3, 1> src(T(observed_x), T(observed_y), T(1));
        Eigen::Matrix<T, 3, 1> dst = H * src;

        // Normalize the projected point
        dst /= dst(2, 0);

        // Compute residuals (reprojection error)
        residuals[0] = dst(0, 0) - T(projected_x);
        residuals[1] = dst(1, 0) - T(projected_y);

        return true;
    }

    double observed_x, observed_y; // source
    double projected_x, projected_y; // dest
};

Eigen::MatrixXd optimizeHomography(const vector<pair<Eigen::Vector2d, Eigen::Vector2d>>& correspondences, Eigen::MatrixXd H) {
    //convert H to array of 8 elems (hard-fixing h33 to 1)
    double h[8];
    for (int i = 0; i < H.rows(); i++) {
        for (int j = 0; j < H.cols(); j++) {
            h[i * H.cols() + j] = H(i, j);
        }
    }

    ceres::Problem problem;

    for (const auto& correspondence : correspondences) {
        // Extract source and destination points from each correspondence
        double observed_x = correspondence.first.x();
        double observed_y = correspondence.first.y();
        double projected_x = correspondence.second.x();
        double projected_y = correspondence.second.y();

        // Create the cost function with the observed and projected points
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<HomographyReprojectionError, 2, 8>(
                new HomographyReprojectionError(observed_x, observed_y, projected_x, projected_y));
        problem.AddResidualBlock(cost_function, nullptr, h);
    }

    // Configure and run the solver
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            H(i, j) = h[i * H.cols() + j];
        }
    }
    H(2, 2) = 1.0;
    return H;
}

void testOnRandomTransforms() {
    Eigen::MatrixXd modelChessboard = generateModelChessboard(3, 4, 1.0);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> tf = generateRandomTransform(10, 10, 10, 100);
    Eigen::MatrixXd hFromTrans = computeHomographyFromTransform(tf);

    Eigen::MatrixXd syntheticImagePoints = hFromTrans * modelChessboard;
    Eigen::MatrixXd syntheticImagePointsC = homToCart(syntheticImagePoints);

    Eigen::MatrixXd hFromSynData = computeHomography(modelChessboard, syntheticImagePointsC);
    Eigen::MatrixXd syntheticHPoints = hFromSynData * modelChessboard;
    Eigen::MatrixXd syntheticHPointsC = homToCart(syntheticHPoints);

    modelChessboard = homToCart(modelChessboard);
    hFromTrans /= hFromTrans(2, 2);
    const vector<pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences = makeCorrespondences(modelChessboard, syntheticImagePointsC);
    hFromSynData = optimizeHomography(correspondences, hFromSynData);
    cout << hFromTrans << endl << endl;
    cout << hFromSynData << endl;
}

void testOnImageData() {
    cv::Mat cvImg = cv::imread("./chessboard_images/upright.png");
    cv::Mat grayImg;
    cv::cvtColor(cvImg, grayImg, cv::COLOR_BGR2GRAY);
    cv::Size numCorners(8, 6);
    int numCornersX = 8;
    int numCornersY = 6;
    
    vector<cv::Point2f> cornersF;
    cv::findChessboardCorners(grayImg, numCorners, cornersF);
    vector<cv::Point2d> corners;
    corners.reserve(cornersF.size());
    
    transform(cornersF.begin(), cornersF.end(), back_inserter(corners), [](const cv::Point2f& pt) -> cv::Point2d {
        return cv::Point2d(pt.x, pt.y);
    });

    Eigen::MatrixXd eigenCorners(3, corners.size());
    eigenCorners = cvToEigen(corners);
    
    Eigen::MatrixXd modelPoints = generateModelChessboard(8, 6, 1.0);
    Eigen::MatrixXd myHomography = computeHomography(modelPoints, eigenCorners);
    assert(modelPoints.rows() == eigenCorners.rows() && modelPoints.cols() == eigenCorners.cols());
    modelPoints = homToCart(modelPoints);
    eigenCorners = homToCart(eigenCorners);
    const vector<pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences = makeCorrespondences(modelPoints, eigenCorners);
    cout << "myHomography" << endl << myHomography << endl << endl;
    myHomography = optimizeHomography(correspondences, myHomography);
    
    vector<cv::Point2d> modelCorners;
    modelCorners = eigenToCv(modelPoints);
    cv::Mat cvHomography = cv::findHomography(modelCorners, corners);

    cout << "myHomography" << endl << myHomography << endl << endl;
    cout << "cvHomography" << endl << cvHomography << endl;
}

int main() {
    //testOnRandomTransforms();
    //testOnImageData();
    k4a_device_configuration_t main_config = get_master_config();
    k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode, main_config.color_resolution);
    getIntrinsic(main_calibration);
    return 0;
}

//H = A * [R R T]
//[R R T] = A^(-1) * H;
//[ R R R T
//  0 0 0 1]