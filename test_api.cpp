//
// Created by ning on 6/25/22.
//
#include <tensorflow/cc/ops/io_ops.h>
#include <tensorflow/cc/ops/parsing_ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/data_flow_ops.h>


#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <fstream>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

int main()
{
    // set up your input paths
    const string pathToGraph = "/home/senius/python/c_python/test/model-10.meta";
    const string checkpointPath = "/home/senius/python/c_python/test/model-10";

    auto session = NewSession(SessionOptions());
    if (session == nullptr)
    {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

    // Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok())
    {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

    // Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if (!status.ok())
    {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

    // Read weights from the saved checkpoint
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<tstring>()() = checkpointPath;
    status = session->Run({{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor},}, {},
                          {graph_def.saver_def().restore_op_name()}, nullptr);
    if (!status.ok())
    {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    cout << 1 << endl;

    const string filename = "/home/senius/python/c_python/test/04t30t00.npy";

    //Read TXT data to array
    float Array[1681*41];
    ifstream is(filename);
    for (int i = 0; i < 1681*41; i++){
        is >> Array[i];
    }
    is.close();

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 41, 41, 41, 1}));
    auto input_tensor_mapped = input_tensor.tensor<float, 5>();

    float *pdata = Array;

    // copying the data into the corresponding tensor
    for (int x = 0; x < 41; ++x)//depth
    {
        for (int y = 0; y < 41; ++y) {
            for (int z = 0; z < 41; ++z) {
                const float *source_value = pdata + x * 1681 + y * 41 + z;
//                input_tensor_mapped(0, x, y, z, 0) = *source_value;
                input_tensor_mapped(0, x, y, z, 0) = 1;
            }
        }
    }

    std::vector<tensorflow::Tensor> finalOutput;
    std::string InputName = "X"; // Your input placeholder's name
    std::string OutputName = "sigmoid"; // Your output placeholder's name
    vector<std::pair<string, Tensor> > inputs;
    inputs.push_back(std::make_pair(InputName, input_tensor));

    // Fill input tensor with your input data
    session->Run(inputs, {OutputName}, {}, &finalOutput);

    auto output_y = finalOutput[0].scalar<float>();
    std::cout << output_y() << "\n";


    return 0;
}