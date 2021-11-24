#ifndef FRICP_H
#define FRICP_H
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "median.h"
#include <limits>
#define SAME_THRESHOLD 1e-6
#include <type_traits>
#include "nanoflann.h"

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
            // unless the result is subnormal
            || std::fabs(x-y) < std::numeric_limits<T>::min();
}
template<int N>
class FRICP
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, N, Eigen::Dynamic> MatrixNX;
    typedef Eigen::Matrix<Scalar, N, N> MatrixNN;
    typedef Eigen::Matrix<Scalar, N+1, N+1> AffineMatrixN;
    typedef Eigen::Transform<Scalar, N, Eigen::Affine> AffineNd;
    typedef Eigen::Matrix<Scalar, N, 1> VectorN;
    typedef nanoflann::KDTreeAdaptor<MatrixNX, N, nanoflann::metric_L2_Simple> KDtree;
    typedef Eigen::Matrix<Scalar, 6, 1> Vector6;
    double test_total_construct_time=.0;
    double test_total_solve_time=.0;
    int test_total_iters=0;

    FRICP(){};
    ~FRICP(){};

public:
    /*
    void findclosestPoints(MatrixNX& X, MatrixNX& Y, std::vector<std::pair<int, int>>& corres){
        /// Build kd-tree
        KDtree kdtree(Y);
        int nPoints = X.cols();

        //Find initial closest point
#pragma omp parallel for
        for (int i = 0; i<nPoints; ++i) {
            int id = kdtree.closest(X.col(i).data());
            corres.push_back(std::pair<int,int>(i, id));
        }
    }
    */
    void findclosestPoints(MatrixNX& X, MatrixNX& Y, std::vector<std::pair<int, int>>& corres){
        /// Build kd-tree
        KDtree kdtree(Y);
        int nPoints = X.cols();   
#pragma omp parallel for
        for (int i = 0; i<nPoints; ++i) {
            Scalar mini_dist;
            int idx = kdtree.closest(X.col(i).data(), mini_dist);
            corres.push_back(std::pair<int,int>(i, idx));
        }
    }
};
#endif


