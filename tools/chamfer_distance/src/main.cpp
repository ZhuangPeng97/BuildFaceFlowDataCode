#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

#include "nanoflann.h"

void get_v(Mesh &mesh, Eigen::MatrixXd &v)
{
    int num_v = mesh.n_vertices();
#pragma omp parallel for
    for (size_t i = 0; i < num_v; ++i)
    {
        v(0, i) = mesh.point(mesh.vertex_handle(i))[0];
        v(1, i) = mesh.point(mesh.vertex_handle(i))[1];
        v(2, i) = mesh.point(mesh.vertex_handle(i))[2];
    }
}

int main(int argc, char const ** argv)
{
    Mesh mesh_src, mesh_tar;
    std::string src_path;
    std::string tar_path;
    std::string csv_path;
    src_path = argv[1];
    tar_path = argv[2];
    csv_path = argv[3];
    OpenMesh::IO::read_mesh(mesh_src, src_path);
    OpenMesh::IO::read_mesh(mesh_tar, tar_path);
    int num_v_src = mesh_src.n_vertices();
    int num_v_tar = mesh_tar.n_vertices();
    // printf("num_v_src, num_v_tar = %d, %d\n", num_v_src, num_v_tar);

    Eigen::MatrixXd v_src(3, num_v_src);
    Eigen::MatrixXd v_tar(3, num_v_tar);
    get_v(mesh_src, v_src);
    get_v(mesh_tar, v_tar);

    Eigen::VectorXi correspondence_src_tar_index(num_v_src);

    KDtree *tree_tar = new KDtree(v_tar);
    for (int i = 0; i < num_v_src; i++)
    {
        double mini_dist;
        Eigen::Vector3d q = v_src.col(i);
        int idx = tree_tar->closest(q.data(), mini_dist);
        correspondence_src_tar_index[i] = idx;
    }

    double src2tar_dist = 0.0;
    for(size_t i = 0; i < num_v_src; i++)
    {
        src2tar_dist += (v_src.col(i) - v_tar.col(correspondence_src_tar_index[i])).norm();
    }

    Eigen::VectorXi correspondence_tar_src_index(num_v_tar);

    KDtree *tree_src = new KDtree(v_src);
    for (int i = 0; i < num_v_tar; i++)
    {
        double mini_dist;
        Eigen::Vector3d q = v_tar.col(i);
        int idx = tree_src->closest(q.data(), mini_dist);
        correspondence_tar_src_index[i] = idx;
    }

    double tar2src_dist = 0.0;
    for(size_t i = 0; i < num_v_tar; i++)
    {
        tar2src_dist += (v_tar.col(i) - v_src.col(correspondence_tar_src_index[i])).norm();
    }

    double avg_src2tar_dist = src2tar_dist / num_v_src;
    double avg_tar2src_dist = tar2src_dist / num_v_tar;
    double cd_distance = avg_src2tar_dist + avg_tar2src_dist;

    std::ofstream oFile;
    oFile.open(csv_path, std::ios::app);
    oFile<<src_path<<","<<src2tar_dist<<","<<tar2src_dist<<","<<avg_src2tar_dist<<","<<avg_tar2src_dist<<","<<cd_distance<<std::endl;
    oFile.close();

    return 1;
}

#if 0
#include <Eigen/Dense>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

#include <nanoflann.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>

//
// https://github.com/jlblancoc/nanoflann/blob/master/examples/utils.h
//
template <typename T>
struct PointCloud
{
    struct Point
    {
        T  x,y,z;
    };

    std::vector<Point>  pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};


//template <typename num_t>
PointCloud<float> get_v(Mesh& mesh)
{
    PointCloud<float> pc;
    pc.pts.resize(mesh.n_vertices());

#pragma omp parallel for
    for (size_t i = 0; i < mesh.n_vertices(); i++)
    {
        pc.pts[i].x = mesh.point(mesh.vertex_handle(i))[0];
        pc.pts[i].y = mesh.point(mesh.vertex_handle(i))[1];
        pc.pts[i].z = mesh.point(mesh.vertex_handle(i))[2];
    }

    return pc;
}


int main()
{

    Mesh mesh_src, mesh_src_merge;
    OpenMesh::IO::read_mesh(mesh_src, "/home/ljh/T_1.obj");
    OpenMesh::IO::read_mesh(mesh_src_merge, "/home/ljh/T.obj");

    PointCloud<float> pc_src = get_v(mesh_src);
    PointCloud<float> pc_src_merge = get_v(mesh_src_merge);
    std::cout << pc_src.pts[0].x << " " << pc_src.pts[0].y << " " << pc_src.pts[0].z << std::endl;

    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, PointCloud<float> > ,
            PointCloud<float>,
            3 /* dim */
            > my_kd_tree_t;

    my_kd_tree_t kd_tree(3 /*dim*/, pc_src, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
    kd_tree.buildIndex();

    const float query_pt[3] = {pc_src_merge.pts[0].x, pc_src_merge.pts[0].y, pc_src_merge.pts[0].z};
    std::cout << "query_pt: " << query_pt << std::endl;

    // ----------------------------------------------------------------
    // radiusSearch(): Perform a search for the points within search_radius
    // ----------------------------------------------------------------
    {
        const float search_radius = pow(0.01, 2);  // attention
        std::vector<std::pair<size_t,float> > ret_matches;

        nanoflann::SearchParams params;
        //params.sorted = false;

        const size_t nMatches = kd_tree.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

        std::cout << "radiusSearch(): radius=" << search_radius << " -> " << nMatches << " matches\n";
        for (size_t i = 0; i < nMatches; i++)
            std::cout << "idx["<< i << "]=" << ret_matches[i].first << " dist["<< i << "]=" << pow(ret_matches[i].second, 0.5) << std::endl;
        std::cout << "\n";
    }
    // ----------------------------------------------------------------
    // knnSearch():  Perform a search for the N closest points
    // ----------------------------------------------------------------
    {
        size_t num_results = 5;
        std::vector<size_t>   ret_index(num_results);
        std::vector<float> out_dist_sqr(num_results);

        num_results = kd_tree.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);

        // In case of less points in the tree than requested:
        ret_index.resize(num_results);
        out_dist_sqr.resize(num_results);

        std::cout << "knnSearch(): num_results=" << num_results << "\n";
        for (size_t i = 0; i < num_results; i++)
            std::cout << "idx["<< i << "]=" << ret_index[i] << " dist["<< i << "]=" << pow(out_dist_sqr[i], 0.5) << std::endl;
        std::cout << "\n";
    }

    return 0;
}
#endif
