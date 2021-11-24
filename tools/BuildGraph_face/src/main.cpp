#include "tools/io_mesh.h"
#include "tools/OmpHelper.h"
#include <iostream>
#include <fstream>
#include <glob.h> // glob(), globfree()
#include "tools/nodeSampler.h"

std::vector<std::string> glob(const std::string &pattern)
{
    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0)
    {
        /*
        globfree(&glob_result);
        std::stringstream ss;
        ss << "glob() failed with return_value " << return_value << std::endl;
        throw std::runtime_error(ss.str());
        */
       std::cout<<"folder is empty"<<std::endl;
       return std::vector<std::string>{};
    }

    // collect all the filenames into a std::list<std::string>
    std::vector<std::string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
    {
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

int main(int argc, char **argv)
{
    std::string save_root = argv[2];

    Mesh src_mesh;
    std::vector<int> src_label;

    std::string src_file = argv[1];

    // Scalar uni_sample_radio = 10;
    // Scalar uni_sample_radio = 8;
    // Scalar uni_sample_radio = 12;
    // Scalar uni_sample_radio = 16;
    // Scalar uni_sample_radio = 6;
    // Scalar uni_sample_radio = 4;
    Scalar uni_sample_radio = 12;

    bool print_each_step_info = true;
    
    int flag = src_file.find_last_of("/");
    std::string prefix = src_file.substr(flag+1, src_file.length()-flag-5);
    read_data(src_file, src_mesh);

    if(src_mesh.n_vertices()==0)
    {
        std::cout<<argv[1]<<" have no vertices!"<<std::endl;
        return 0;
    }

    svr::nodeSampler src_sample_nodes;
    Scalar sample_radius = src_sample_nodes.sampleAndconstuct(src_mesh, uni_sample_radio, svr::nodeSampler::X_AXIS);
    std::string out_node = save_root + prefix + "_";
    
    src_sample_nodes.print_nodes(src_mesh, out_node);
    return 0;
}
