#include <iostream>
#include "io_pc.h"
#include "FRICP.h"

int main(int argc, char const ** argv)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;
    std::string file_source;
    std::string file_target;
    std::string file_init = "./data/";
    std::string res_trans_path;
    std::string out_path;
    enum Method{ICP, AA_ICP, FICP, RICP, PPL, RPPL, SparseICP, SICPPPL} method=RICP;
    struct test_st
    {
        double E;
        Vertices points;
        Vertices normals;
    };

    if(argc == 5)
    {
        file_target = argv[1];
        file_source = argv[2];
        out_path = argv[3];
        method = Method(std::stoi(argv[4]));
    }
    else if(argc==4)
    {
        file_target = argv[1];
        file_source = argv[2];
        out_path = argv[3];
    }
    else
    {
        std::cout << "Usage: target.ply source.ply out_path <Method>" << std::endl;
        std::cout << "Method :\n"
                  << "0: ICP\n1: AA-ICP\n2: Our Fast ICP\n3: Our Robust ICP\n4: ICP Point-to-plane\n"
                  << "5: Our Robust ICP point to plane\n6: Sparse ICP\n7: Sparse ICP point to plane" << std::endl;
        exit(0);
    }
    int dim = 3;


    //--- Model that will be rigidly transformed
    Vertices vertices_source, normal_source, src_vert_colors;
    read_file(vertices_source, normal_source, src_vert_colors, file_source);
    std::cout << "source: " << vertices_source.rows() << "x" << vertices_source.cols() << std::endl;

    //--- Model that source will be aligned to
    Vertices vertices_target, normal_target, tar_vert_colors;
    read_file(vertices_target, normal_target, tar_vert_colors, file_target);
    std::cout << "target: " << vertices_target.rows() << "x" << vertices_target.cols() << std::endl;
    std::cout << "normal_target = " << normal_target.col(0).transpose() << std::endl;
    std::cout << "color_target = " << tar_vert_colors.col(0).transpose() << std::endl;

    std::cout << "source_pc = " << vertices_source.col(0).transpose() << std::endl;
    Vertices new_vertices_source = 2 * vertices_source;
    Vertices new_normals_source = 2 * normal_source;

//    test_st *t1 = new test_st;
//    t1->E = 1.0;
//    t1->points = vertices_source;
//    t1->normals = normal_source;
//    test_st *t2 = new test_st;
//    t2->E = 2.0;
//    t2->points = new_vertices_source;
//    t2->normals = new_normals_source;

//    std::cout << "t1_e = " << t1->E << " p = " << t1->points(0,0) << " n = " << t1->normals(0,0) << std::endl;
//    std::cout << "t2_e = " << t2->E << " p = " << t2->points(0,0) << " n = " << t2->normals(0,0) << std::endl;

//    test_st *t3 = t1;
//    t1 = t2;
//    t2 = t3;
//    std::swap(t1, t2);

//    std::cout << "t1_e = " << t1->E << " p = " << t1->points(0,0) << " n = " << t1->normals(0,0) << std::endl;
//    std::cout << "t2_e = " << t2->E << " p = " << t2->points(0,0) << " n = " << t2->normals(0,0) << std::endl;


//    ///--- Execute registration
//    std::cout << "begin registration..." << std::endl;
    FRICP<3> fricp;
    std::vector<std::pair<int,int>> corres;
    fricp.findclosestPoints(vertices_source, vertices_target, corres);

    double src2tar_dist = 0.0;
    for(size_t i = 0; i < corres.size(); i++)
    {
        src2tar_dist += (vertices_source.col(corres[i].first) - vertices_target.col(corres[i].second)).norm();
    }
    std::vector<std::pair<int,int>> corres1;
    fricp.findclosestPoints(vertices_target, vertices_source, corres1);
    double tar2src_dist = 0.0;
    for(size_t i = 0; i < corres1.size(); i++)
    {
        tar2src_dist += (vertices_target.col(corres1[i].first) - vertices_source.col(corres1[i].second)).norm();
    }

    double avg_src2tar_dist = src2tar_dist / corres.size();
    double avg_tar2src_dist = tar2src_dist / corres1.size();
    double cd_distance = avg_src2tar_dist + avg_tar2src_dist;
    std::ofstream oFile;
    oFile.open(out_path, std::ios::app|std::ios::trunc);
    oFile<<file_source<<","<<src2tar_dist<<","<<tar2src_dist<<","<<avg_src2tar_dist<<","<<avg_tar2src_dist<<","<<cd_distance<<std::endl;
    oFile.close();
    return 0;
}
