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



Eigen::MatrixXd getCameraIntrinsicMatrix() {
    try {
        k4a::device device = k4a::device::open(K4A_DEVICE_DEFAULT);
        k4a::calibration calibration = device.get_calibration(K4A_DEPTH_MODE_NFOV_UNBINNED, K4A_COLOR_RESOLUTION_1080P);

        const k4a_calibration_intrinsic_parameters_t::_param &i = calibration.color_camera_calibration.intrinsics.parameters.param;
        Eigen::MatrixXd camera_matrix(3, 3);
        camera_matrix << i.fx, 0, i.cx,
                         0, i.fy, i.cy,
                         0, 0, 1;

        device.close();
        return camera_matrix;
    } catch (const std::exception& e) {
        std::cerr << "Failed to open device or get calibration: " << e.what() << std::endl;
        return Eigen::MatrixXd::Zero(3, 3);
    }
}


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

//these points should be cartesian
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

    outH /= outH(2,2); //normalize bottom right corner to 1

    return outH;
}

//gets a rigid transform given a homography and camera intrinsic matrix
Eigen::Matrix<double, 7, 1> getRigidTransform(const Eigen::MatrixXd& homography, const Eigen::MatrixXd& cameraIntrinsics) {
    // H = A[r1|r2|t]
    Eigen::MatrixXd rrt = cameraIntrinsics.inverse() * homography;
    Eigen::Vector3d r1 = rrt.col(0).head<3>();
    Eigen::Vector3d r2 = rrt.col(1).head<3>();
    Eigen::Vector3d t = rrt.col(2).head<3>();

    // r1 and r2 should be orthogonal already in theory, but in practice they could be slightly off due to noise.
    // we need to normalize them anyway, so we might as well apply Gram-Schmidt ensuring they're both orthonormal

    r1.normalize();

    Eigen::Vector3d proj_r2_on_r1 = (r2.dot(r1) / r1.dot(r1)) * r1;
    r2 = r2 - proj_r2_on_r1;
    r2.normalize();

    // Compute r3 as cross product
    Eigen::Vector3d r3 = r1.cross(r2);

    // Reconstruct rotation matrix
    Eigen::Matrix3d rot;
    rot.col(0) = r1;
    rot.col(1) = r2;
    rot.col(2) = r3;

    // Convert rotation matrix to quaternion
    Eigen::Quaterniond quat(rot);

    // Create the rigid transformation matrix
    Eigen::Matrix<double, 7, 1> rigidTransform;
    rigidTransform.head<4>() = quat.coeffs();
    rigidTransform.tail<3>() = t;
    return rigidTransform;
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

Eigen::MatrixXd cartToHom(Eigen::MatrixXd srcPts) {
    Eigen::MatrixXd dstPts(srcPts.rows() + 1, srcPts.cols());
    dstPts.block(0,0,srcPts.rows(), srcPts.cols()) = srcPts;
    for(int i = 0; i < srcPts.cols(); i++) {
        dstPts(srcPts.rows(), i) = 1;
    }
    return dstPts;
}

Eigen::MatrixXd zeroPadLastRow(Eigen::MatrixXd srcPts) {
    Eigen::MatrixXd dstPts(srcPts.rows() + 1, srcPts.cols());
    dstPts.block(0,0,srcPts.rows(), srcPts.cols()) = srcPts;
    for(int i = 0; i < srcPts.cols(); i++) {
        dstPts(srcPts.rows(), i) = 0;
    }
    return dstPts;
}

// struct HomographyReprojectionError {
//     HomographyReprojectionError(double observed_x, double observed_y, double projected_x, double projected_y)
//         : observed_x(observed_x), observed_y(observed_y), projected_x(projected_x), projected_y(projected_y) {}

//     template <typename T>
//     bool operator()(const T* const homography, T* residuals) const {
//         // Reconstruct the homography matrix H from the parameter vector
//         Eigen::Matrix<T, 3, 3> H;

//         H << homography[0], homography[1], homography[2],
//              homography[3], homography[4], homography[5],
//              homography[6], homography[7], T(1);

//         // Apply homography to the observed point
//         Eigen::Matrix<T, 3, 1> src(T(observed_x), T(observed_y), T(1));
//         Eigen::Matrix<T, 3, 1> dst = H * src;

//         // Normalize the projected point
//         dst /= dst(2, 0);

//         // Compute residuals (reprojection error)
//         residuals[0] = dst(0, 0) - T(projected_x);
//         residuals[1] = dst(1, 0) - T(projected_y);

//         return true;
//     }

//     double observed_x, observed_y; // source
//     double projected_x, projected_y; // dest
// };

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homToCart(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> srcPts) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dstPts(srcPts.rows() - 1, srcPts.cols());
    for(int i = 0; i < srcPts.rows() - 1; i++)
        dstPts.row(i) = srcPts.row(i).array() / srcPts.row(srcPts.rows() - 1).array();
    return dstPts;
}

struct ZhangsOptimization {
    Eigen::Matrix<double, 2, Eigen::Dynamic> _imagePoints;
    Eigen::Matrix<double, 4, Eigen::Dynamic> _modelPoints;
    Eigen::Matrix<double, 3, 3> _cameraIntrinsics;

    ZhangsOptimization(
        const Eigen::Matrix<double, 2, Eigen::Dynamic>& imagePoints,
        const Eigen::Matrix<double, 4, Eigen::Dynamic>& modelPoints, // XYZW = [X, Y, 0, 1.0]
        const Eigen::Matrix<double, 3, 3>& cameraIntrinsics)
        : _imagePoints(imagePoints), _modelPoints(modelPoints), //model points should be 3D homogeneous. image points are 2d cartesian
        _cameraIntrinsics(cameraIntrinsics) {}

    template<typename T>
    bool operator()(const T* const transform, T* residuals) const {
        Eigen::Quaternion<T> quat(transform[0], transform[1], transform[2], transform[3]);
        Eigen::Matrix<T, 3, 3> R = quat.toRotationMatrix();
        Eigen::Matrix<T, 3, 4> extrinsic;
        extrinsic << R, Eigen::Matrix<T, 3, 1>(transform[4], transform[5], transform[6]);
        Eigen::Matrix<T, 3, 3> cameraIntrinsicsT = _cameraIntrinsics.template cast<T>();
        for (int i = 0; i < _modelPoints.cols(); ++i) {
            Eigen::Matrix<T, 4, 1> modelPoint = _modelPoints.col(i).template cast<T>();
            Eigen::Matrix<T, 3, 1> projectedPoint = cameraIntrinsicsT * (extrinsic * modelPoint); // 3x3 times (3x4 times 4x1) = 3x1
            projectedPoint /= projectedPoint(2, 0); // Convert from homogeneous to Cartesian coordinates
            //cout << "Projected point = " << projectedPoint << endl;
            int idx = i * 2; // 2 residuals per point (x, y)
            residuals[idx] = projectedPoint(0) - T(_imagePoints(0, i));
            residuals[idx + 1] = projectedPoint(1) - T(_imagePoints(1, i));
        }
        return true;
    }
};

Eigen::Matrix<double, 7, 1> optimizeHomography(
    const Eigen::Matrix<double, 2, Eigen::Dynamic>& imagePoints, //simple X, Y
    const Eigen::Matrix<double, 4, Eigen::Dynamic>& modelPoints, //[X, Y, 0, 1.0]
    const Eigen::Matrix<double, 3, 3>& cameraIntrinsics,
    const Eigen::Matrix<double, 7, 1>& initialTransform) {
    double rt[7];
    for(int i = 0; i < 7; i++) rt[i] = initialTransform(i,0);

    ceres::Problem problem;
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ZhangsOptimization, ceres::DYNAMIC, 7>(
        new ZhangsOptimization(imagePoints, modelPoints, cameraIntrinsics), imagePoints.cols() * 2);
    problem.AddResidualBlock(cost_function, nullptr, rt);

    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    Eigen::Matrix<double, 7, 1> optimizedTransform;
    for(int i = 0; i < 7; i++) optimizedTransform(i,0) = rt[i];
    
    return optimizedTransform;
}


// Eigen::MatrixXd optimizeHomography(const vector<pair<Eigen::Vector2d, Eigen::Vector2d>>& correspondences, Eigen::MatrixXd H) {
//     //convert H to array of 8 elems (hard-fixing h33 to 1)
//     double h[8];
//     for (int i = 0; i < H.rows(); i++) {
//         for (int j = 0; j < H.cols(); j++) {
//             h[i * H.cols() + j] = H(i, j);
//         }
//     }

//     ceres::Problem problem;

//     for (const auto& correspondence : correspondences) {
//         // Extract source and destination points from each correspondence
//         double observed_x = correspondence.first.x();
//         double observed_y = correspondence.first.y();
//         double projected_x = correspondence.second.x();
//         double projected_y = correspondence.second.y();

//         // Create the cost function with the observed and projected points
//         ceres::CostFunction* cost_function =
//             new ceres::AutoDiffCostFunction<HomographyReprojectionError, 2, 8>(
//                 new HomographyReprojectionError(observed_x, observed_y, projected_x, projected_y));
//         problem.AddResidualBlock(cost_function, nullptr, h);
//     }

//     // Configure and run the solver
//     ceres::Solver::Options options;
//     options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//     options.linear_solver_type = ceres::DENSE_QR;
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);

//     std::cout << summary.FullReport() << "\n";
    
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++) {
//             H(i, j) = h[i * H.cols() + j];
//         }
//     }
//     H(2, 2) = 1.0;
//     return H;
// }

void testOnRandomTransforms() {
    Eigen::MatrixXd modelChessboard = generateModelChessboard(3, 4, 1.0);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> tf = generateRandomTransform(10, 10, 10, 100);
    Eigen::MatrixXd hFromTrans = computeHomographyFromTransform(tf);

    Eigen::MatrixXd syntheticImagePoints = hFromTrans * modelChessboard;
    Eigen::MatrixXd cartesianImagePoints = homToCart(syntheticImagePoints);

    Eigen::MatrixXd cameraIntrinsics = getCameraIntrinsicMatrix();
    // ZhangsOptimization<double> zo(syntheticImagePoints, modelChessboard, cameraIntrinsics);
    Eigen::MatrixXd H = computeHomography(modelChessboard, cartesianImagePoints);
    cout << "H" << endl << H << endl;
    hFromTrans /= hFromTrans(2, 2);
    cout << "HFromTrans" <<endl << hFromTrans << endl;

    Eigen::Matrix<double, 7, 1> realTransform = getRigidTransform(hFromTrans, Eigen::MatrixXd::Identity(3, 3)); //transform with no camera distortion

    Eigen::Matrix<double, 7, 1> rigidTransform = getRigidTransform(H, Eigen::MatrixXd::Identity(3, 3));

    modelChessboard = homToCart(modelChessboard);
    modelChessboard = zeroPadLastRow(modelChessboard);
    modelChessboard = cartToHom(modelChessboard);
    
    
    cout << "First Rigid Transform" << endl << rigidTransform << endl;
    rigidTransform = optimizeHomography(cartesianImagePoints, modelChessboard, Eigen::MatrixXd::Identity(3, 3), rigidTransform);
    cout << "Rigid Transform" << endl << rigidTransform << endl;
    
}

/*
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
*/

void addPoints(std::vector<cv::Point>& points, const std::vector<cv::Point>& newPoints);
cv::Mat createBlankImageWithPoints(const std::vector<cv::Point>& points, const cv::Size& imageSize);
void displayImage(const cv::Mat& image, const std::string& windowName);

void addPoints(std::vector<cv::Point>& points, const std::vector<cv::Point>& newPoints) {
    points.insert(points.end(), newPoints.begin(), newPoints.end());
}

cv::Mat createBlankImageWithPoints(vector<cv::Point2d> points, const cv::Size& imageSize) {
    cv::Mat blankImage = cv::Mat::zeros(imageSize, CV_8UC3);
    for (cv::Point2d point : points) {
        cv::circle(blankImage, point, 5, cv::Scalar(0, 255, 0), -1); // green color
    }
    return blankImage;
}

void displayImage(const cv::Mat& image, const std::string& windowName) {
    cv::imshow(windowName, image);
    cv::waitKey(0);
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
    
    //convert float corners (from openCV) to double corners
    transform(cornersF.begin(), cornersF.end(), back_inserter(corners), [](const cv::Point2f& pt) -> cv::Point2d {
        return cv::Point2d(pt.x, pt.y);
    });

    Eigen::MatrixXd eigenCorners(3, corners.size());
    eigenCorners = cvToEigen(corners);
    
    Eigen::MatrixXd modelPoints = generateModelChessboard(8, 6, 1.0);
    Eigen::MatrixXd H = computeHomography(modelPoints, eigenCorners);

    

    assert(modelPoints.rows() == eigenCorners.rows() && modelPoints.cols() == eigenCorners.cols());

    modelPoints = homToCart(modelPoints); //chop off W row of 1.0
    modelPoints = zeroPadLastRow(modelPoints); //add row of 0s as the Z coordinates
    modelPoints = cartToHom(modelPoints); //add W row of 1.0 back.

    eigenCorners = homToCart(eigenCorners); //make image points 2d cartesian

    Eigen::Matrix<double, 3, 3> A = getCameraIntrinsicMatrix(); //must be plugged into camera
 
    Eigen::Matrix<double, 7, 1> rt = getRigidTransform(H, A);


    //rigidTransform = the actual rigidTransform
    cout << "Old RT" << endl << rt << endl;
    rt = optimizeHomography(eigenCorners, modelPoints, A, rt);
    cout << "NEw RT" << endl << rt << endl;
    Eigen::Quaterniond quat(rt[0], rt[1], rt[2], rt[3]); //w, x, y, z
    Eigen::Vector3d translation(rt[4], rt[5], rt[6]);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    Eigen::Matrix4d rigidTransform = Eigen::Matrix4d::Identity(); 
    rigidTransform.block<3,3>(0,0) = rot; // Top-left 3x3 block is the rotation matrix
    rigidTransform.block<3,1>(0,3) = translation; // Top-right 3x1 block is the translation vector
    // The last row is already [0, 0, 0, 1] due to the initialization with Identity()

    //cout << "Projected points: " << endl;
    Eigen::MatrixXd projectedCorners = (rigidTransform * modelPoints);
    projectedCorners = homToCart(projectedCorners);
    projectedCorners = A * projectedCorners;
    // for (int i = 0; i < projectedCorners.cols(); i++) {
    //     cout << projectedCorners(0, i) << ", " << projectedCorners(1, i) << endl;
    // }
    // cout << "Image points: " << endl;
    // for (int i = 0; i < eigenCorners.cols(); i++) {
    //     cout << eigenCorners(0, i) << ", " << eigenCorners(1, i) << endl;
    // }



    //cv test
    const vector<cv::Point2d> cvCorners = eigenToCv(eigenCorners);
    const vector<cv::Point2d> cvProjections = eigenToCv(projectedCorners);
    // Placeholder for image size - replace with actual image loading if necessary
    cv::Size imageSize(1920, 1080); // Assuming a 500x500 image size for demonstration

    cv::Mat imageWithPoints = createBlankImageWithPoints(cvCorners, imageSize);
    cv::Mat projectedImage = createBlankImageWithPoints(cvProjections, imageSize);
    displayImage(imageWithPoints, "Image Points");
    displayImage(projectedImage, "Projected Points");


    // vector<cv::Point2d> modelCorners;
    // modelCorners = eigenToCv(modelPoints);
    // cv::Mat cvHomography = cv::findHomography(modelCorners, corners);

    // cout << "myHomography" << endl << myHomography << endl << endl;
    // cout << "cvHomography" << endl << cvHomography << endl;
}

/*
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
*/

// Function prototypes


int main() {


    //testOnRandomTransforms();
    testOnImageData();
    
    return 0;
}


//H = A * [R R T]
//[R R T] = A^(-1) * H;
//[ R R R T
//  0 0 0 1]