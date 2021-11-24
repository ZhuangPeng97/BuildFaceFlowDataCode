//#pragma once
#include "nodeSampler.h"
#include "tools.h"

//#include <igl/heat_geodesics.h>
//#include <igl/read_triangle_mesh.h>
#include "geodesic/geodesic_algorithm_exact.h"

namespace svr
{
    //	Define helper functions
    static auto square = [](const Scalar argu) { return argu * argu; };
    static auto cube = [](const Scalar argu) { return argu * argu * argu; };
    static auto max = [](const Scalar lhs, const Scalar rhs) { return lhs > rhs ? lhs : rhs; };

    //------------------------------------------------------------------------
    //	Node Sampling based on geodesic distance metric
    //
    //	Note that this member function samples nodes along some axis.
    //	Each node is not covered by any other node. And distance between each
    //	pair of nodes is at least sampling radius.
    //------------------------------------------------------------------------
    // heat method
   Scalar nodeSampler::sample(Mesh &mesh, Scalar sampleRadiusRatio, sampleAxis axis)
    {
//        //	Save numbers of vertex and edge
//        m_meshVertexNum = mesh.n_vertices();
//        m_meshEdgeNum = mesh.n_edges();
//        m_mesh = & mesh;

//        //	Calculate average edge length of bound mesh
//        for (size_t i = 0; i < m_meshEdgeNum; ++i)
//        {
//            OpenMesh::EdgeHandle eh = mesh.edge_handle(i);
//            Scalar edgeLen = mesh.calc_edge_length(eh);
//            m_averageEdgeLen += edgeLen;
//        }
//        m_averageEdgeLen /= m_meshEdgeNum;

//        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
//        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

//        //	Reorder mesh vertex along axis
//        std::vector<size_t> vertexReorderedAlongAxis(m_meshVertexNum);
//        size_t vertexIdx = 0;
//        std::generate(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&vertexIdx]() -> size_t { return vertexIdx++; });
//        std::sort(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&mesh, axis](const size_t &lhs, const size_t &rhs) -> bool {
//            size_t lhsIdx = lhs;
//            size_t rhsIdx = rhs;
//            OpenMesh::VertexHandle vhl = mesh.vertex_handle(lhsIdx);
//            OpenMesh::VertexHandle vhr = mesh.vertex_handle(rhsIdx);
//            Mesh::Point vl = mesh.point(vhl);
//            Mesh::Point vr = mesh.point(vhr);
//            return vl[axis] > vr[axis];
//        });

//        //	Sample nodes using radius of m_sampleRadius
//        size_t firstVertexIdx = vertexReorderedAlongAxis[0];
//        VectorX geoDistVector(m_meshVertexNum);
//        geoDistVector.setZero();

//        igl::HeatGeodesicsData<Scalar> data;
//        MatrixXX V1;
//        Eigen::MatrixXi F1;
//        data.use_intrinsic_delaunay = true;
//        Mesh2VF(mesh, V1, F1);
//        igl::heat_geodesics_precompute(V1, F1, data);

//        igl::heat_geodesics_solve(data, (Eigen::MatrixXi(1,1)<<firstVertexIdx).finished(), geoDistVector);
//        m_geoDistContainer.push_back(geoDistVector/2.0);
//        m_nodeContainer.emplace_back(0, firstVertexIdx);
//        VertexNodeIdx.resize(m_meshVertexNum, -1);

//        for (auto &vertexIdx : vertexReorderedAlongAxis)
//        {
//            bool IsNode = true;
//            for (size_t k = 0; k < m_geoDistContainer.size(); ++k)
//            {
//                Scalar dist = m_geoDistContainer.at(k)(vertexIdx);
//                if (dist < m_sampleRadius)
//                {
//                    IsNode = false;
//                    break;
//                }
//            }
//            if (IsNode)
//            {
//                geoDistVector.setZero();
//                igl::heat_geodesics_solve(data, (Eigen::MatrixXi(1,1)<<vertexIdx).finished(), geoDistVector);
//                m_geoDistContainer.push_back(geoDistVector/2.0);
//                m_nodeContainer.emplace_back(m_geoDistContainer.size() - 1, vertexIdx);
//                VertexNodeIdx[vertexIdx] = m_geoDistContainer.size()-1;
//            }
//        }

//        std::cout << "m_radius = " << m_sampleRadius << std::endl;
//        return m_sampleRadius;
    }


    // Local geodesic calculation
    Scalar nodeSampler::sampleAndconstuct(Mesh &mesh, Scalar sampleRadiusRatio, sampleAxis axis)
    {
        //	Save numbers of vertex and edge
        m_meshVertexNum = mesh.n_vertices();
        m_meshEdgeNum = mesh.n_edges();
        m_mesh = & mesh;

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshEdgeNum; ++i)
        {
            OpenMesh::EdgeHandle eh = mesh.edge_handle(i);
            Scalar edgeLen = mesh.calc_edge_length(eh);
            m_averageEdgeLen += edgeLen;
        }
        m_averageEdgeLen /= m_meshEdgeNum;

        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

        //	Reorder mesh vertex along axis
        std::vector<size_t> vertexReorderedAlongAxis(m_meshVertexNum);
        size_t vertexIdx = 0;
        std::generate(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&vertexIdx]() -> size_t { return vertexIdx++; });
        std::sort(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&mesh, axis](const size_t &lhs, const size_t &rhs) -> bool {
            size_t lhsIdx = lhs;
            size_t rhsIdx = rhs;
            OpenMesh::VertexHandle vhl = mesh.vertex_handle(lhsIdx);
            OpenMesh::VertexHandle vhr = mesh.vertex_handle(rhsIdx);
            Mesh::Point vl = mesh.point(vhl);
            Mesh::Point vr = mesh.point(vhr);
            return vl[axis] > vr[axis];
        });

        //	Sample nodes using radius of m_sampleRadius
        size_t firstVertexIdx = vertexReorderedAlongAxis[0];
        VertexNodeIdx.resize(m_meshVertexNum);
        VertexNodeIdx.setConstant(-1);
        VertexNodeIdx[firstVertexIdx] = 0;
        size_t cur_node_idx = 0;

        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum = VectorX::Zero(m_meshVertexNum);

        for (auto &vertexIdx : vertexReorderedAlongAxis)
        {
            if(VertexNodeIdx[vertexIdx] < 0 && m_vertexGraph.at(vertexIdx).empty())
            {
                m_nodeContainer.emplace_back(cur_node_idx, vertexIdx);
                VertexNodeIdx[vertexIdx] = cur_node_idx;

                std::vector<size_t> neighbor_verts;
                geodesic::GeodesicAlgorithmExact geoalg(&mesh, vertexIdx, m_sampleRadius, 5);
                geoalg.propagate(vertexIdx, neighbor_verts);
                for(size_t i = 0; i < neighbor_verts.size(); i++)
                {
                    int neighIdx = neighbor_verts[i];
                    Scalar geodist = mesh.data(mesh.vertex_handle(neighIdx)).geodesic_distance;

                    if(geodist < m_sampleRadius)
                    {
                        Scalar weight = std::pow(1-std::pow(geodist/m_sampleRadius, 2), 3);
                        m_vertexGraph.at(neighIdx).emplace(std::pair<int, Scalar>(cur_node_idx, weight));
                        weight_sum[neighIdx] += weight;
                    }
                }
                cur_node_idx++;
            }
        }
        m_nodeGraph.resize(cur_node_idx);
        for (auto &vertexIdx : vertexReorderedAlongAxis)
        {
            for(auto &node: m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = node.first;
                for(auto &neighNode: m_vertexGraph[vertexIdx])
                {
                    size_t neighNodeIdx = neighNode.first;
                    if(nodeIdx != neighNodeIdx)
                    {
                        m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(neighNodeIdx, 1.0));
                    }
                }
                m_vertexGraph.at(vertexIdx).at(nodeIdx) /= weight_sum[vertexIdx];
            }
        }
        return m_sampleRadius;
    }

    void nodeSampler::print_nodes(Mesh & mesh, std::string file_path)
    {
        std::string namev = file_path + "nodes.obj";
        std::ofstream out1(namev);
        //std::cout << "print nodes to " << name << std::endl;
        for (size_t i = 0; i < m_nodeContainer.size(); i++)
        {
            int vexid = m_nodeContainer[i].second;
            out1 << "v " << mesh.point(mesh.vertex_handle(vexid))[0] << " " << mesh.point(mesh.vertex_handle(vexid))[1]
                << " " << mesh.point(mesh.vertex_handle(vexid))[2] << std::endl;
        }
        Eigen::VectorXi nonzero_num = Eigen::VectorXi::Zero(m_nodeContainer.size());
        for (size_t nodeIdx = 0; nodeIdx < m_nodeContainer.size(); ++nodeIdx)
        {
            for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
            {
                size_t neighborIdx = eachNeighbor.first;
                out1 << "l " << nodeIdx+1 << " " << neighborIdx+1 << std::endl;
            }
            nonzero_num[nodeIdx] = m_nodeGraph[nodeIdx].size();
        }
        // std::cout << "node neighbor min = " << nonzero_num.minCoeff() << " max = "
        //           << nonzero_num.maxCoeff() << " average = " << nonzero_num.mean() << std::endl;
        out1.close();
        std::string namee = file_path + "edges.txt";
        std::ofstream out2(namee);
        std::string namevtxt = file_path + "nodes.txt";
        std::ofstream out3(namevtxt);
        for (size_t nodeIdx = 0; nodeIdx < m_nodeContainer.size(); ++nodeIdx)
        {
            size_t vIdx0 = getNodeVertexIdx(nodeIdx);
            out3 << vIdx0 <<std::endl;
            for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
            {
                size_t neighborIdx = eachNeighbor.first;
                Scalar flag = eachNeighbor.second;
                size_t vIdx1 = getNodeVertexIdx(neighborIdx);
                out2 << nodeIdx << " " << neighborIdx << " " << vIdx0 << " " << vIdx1 << " " << flag << std::endl;
            }
        }
        out2.close();
        out3.close();
    }
}
