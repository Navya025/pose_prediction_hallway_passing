#include <ceres/ceres.h>        
#include <ceres/rotation.h>   
#include <Eigen/Core>           
#include <Eigen/Dense>         
#include <glog/logging.h>         

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

void OptimizeHomography(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& correspondences, Eigen::MatrixXd H) {
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
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

}