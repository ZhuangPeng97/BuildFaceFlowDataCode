#include "tools/io_mesh.h"
#include "tools/OmpHelper.h"
#include "NonRigidreg.h"
#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
    Mesh src_mesh;
    Mesh tar_mesh;
    std::string src_file;
    std::string tar_file;
    std::string out_file, outpath;
    std::string landmark_file;
    std::string res_name;
    bool invert;
    RegParas paras;

    if(argc==5)
    {
        src_file = argv[1];
        tar_file = argv[2];
        outpath = argv[3];
        res_name = argv[4];
    }
    else if(argc==6)
    {
        src_file = argv[1];
        tar_file = argv[2];
        outpath = argv[3];
        res_name = argv[4];
        landmark_file = argv[5];
        paras.use_landmark = true;
    }
    else
    {
        std::cout << "Usage: <srcFile> <tarFile> <outPath>\n    or <srcFile> <tarFile> <outPath> <landmarkFile>" << std::endl;
        exit(0);
    }

    // Setting paras
    paras.alpha = 100.0;
    paras.beta =  100.0;
    paras.gamma = 1e8;

    paras.use_distance_reject = false;
    paras.distance_threshold = 0.05;
    paras.use_normal_reject = false;
    paras.normal_threshold = M_PI/3;
    paras.use_Dynamic_nu = false;
    paras.uni_sample_radio = 5;

    paras.out_gt_file = outpath + "_res.txt";
    paras.print_each_step_info = false;
    paras.out_each_step_info = outpath + res_name;
//    paras.out_res_name = res_name;
    out_file = paras.out_each_step_info + ".obj";
    
    //out_file = outpath;
    read_data(src_file, src_mesh);
    read_data(tar_file, tar_mesh);
    if(src_mesh.n_vertices()==0 || tar_mesh.n_vertices()==0)
        exit(0);

    if(src_mesh.n_vertices() != tar_mesh.n_vertices())
        paras.calc_gt_err = false;

    if(paras.use_landmark)
    {
        read_landmark(landmark_file.c_str(), paras.landmark_src, paras.landmark_tar);
    }

    double scale = mesh_scaling(src_mesh, tar_mesh);

    NonRigidreg* reg;
    reg = new NonRigidreg;

    Timer time;
    // std::cout << "\nrigid registration to initial..." << std::endl;
    Timer::EventID time1 = time.get_time();
    reg->rigid_init(src_mesh, tar_mesh, paras);
    reg->DoRigid();
    Timer::EventID time2 = time.get_time();
    // std::cout << "rgid registration... " << std::endl;
    // non-rigid initialize
    // std::cout << "non-rigid registration to initial..." << std::endl;
    Timer::EventID time3 = time.get_time();
    reg->Initialize();
    Timer::EventID time4 = time.get_time();
    reg->pars_.non_rigid_init_time = time.elapsed_time(time3, time4);
    // std::cout << "non-rigid registration... " << std::endl;
    reg->DoNonRigid();
    Timer::EventID time5 = time.get_time();

    std::vector<float> loss;
    std::vector<Eigen::Matrix<float, 3, 1> > each;
    loss = reg->pars_.each_energys;
    each = reg->pars_.each_term_energy;
//    for(int i =0; i< loss.size(); i++)
//    std::cout << "loss:" << loss[loss.size()-1] <<std::endl;
//    for(int i = 0; i < each[0].size(); i++)
//	std::cout << "each:" << each[each.size()-1][i] <<std::endl;
    // std::ofstream outfile(outpath + "loss.txt", std::ios::app);

    // outfile  << src_file << " " << tar_file << " " <<  out_file << " " << loss[loss.size()-1] << " " << each[each.size()-1][0] << " " << each[each.size()-1][1] << " " << each[each.size()-1][2] << std::endl;
    // std::cout << "Registration done!\nrigid_init time : "
    //           << time.elapsed_time(time1, time2) << " s \trigid-reg run time = " << time.elapsed_time(time2, time3)
    //           << " s \nnon-rigid init time = "
    //           << time.elapsed_time(time3, time4) << " s \tnon-rigid run time = "
    //           << time.elapsed_time(time4, time5) << " s\n" << std::endl;
    write_data(out_file.c_str(), src_mesh, scale);
    //std::cout<< "write result to " << out_file << std::endl;
    delete reg;

    return 0;
}
