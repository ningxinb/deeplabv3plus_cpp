#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include <tensorflow/cc/ops/io_ops.h>
#include "tensorflow/core/framework/tensor.h"
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace tensorflow;
using namespace cv;
//using namespace tensorflow::ops;

void CVMat_to_Tensor(Mat img, Tensor* output_tensor, int input_rows, int input_cols) {
    resize(img, img, Size(input_cols, input_rows));
    img.convertTo(img, CV_32FC3);
    float * p = output_tensor->flat<float>().data();

    Mat tempMat(input_rows, input_cols, CV_32FC3, p);
    img.convertTo(tempMat, CV_32FC3);
}


void tensor2Mat(Tensor &t, Mat &image, int input_rows, int input_cols) {
    float *p = t.flat<float>().data();
    image = Mat(input_rows, input_cols, CV_32FC1, p);
    image.convertTo(image, CV_8UC1);
}

template<class T>
inline size_t argmax(T first, T last)
{
    return std::distance(first, std::max_element(first, last));
}


int main()
{
    /*----------------setting------------------*/
    const string graph_path = "/home/ning/ws/models/tensorflow-deeplab-v3-plus/model/pb/frozen_model.pb";
    const string image_path = "/home/ning/Pictures/people.jpg";
    const string label_colors_path = "./pascal.png";
    int input_height = 513;
    int input_width = 513;
    int class_nums = 21;
    string input_tensor_name = "IteratorGetNext:0";
    vector<string> output_nodes{"softmax_tensor"};

    /*-----------------create session-------------*/
    GraphDef graph_def;
    SessionOptions session_options;
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
//    session_options.config.mutable_gpu_options()->set_visible_device_list("0");
    Session* session = NewSession(session_options);
    if(session == nullptr) {
        cout << "Could not create Tensorflow session." << endl;
    }

    /*------------------read model from pb file------------*/
    Status status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
    if(!status.ok()) {
        cout << "Error reading graph, failed to load pb model!" << endl;
        return -1;
    }
//    string my_device = "/gpu:" + device;
    status = session->Create(graph_def);
    if(!status.ok()) {
        cout << "Error creating graph" << endl;
        return -1;
    }
    cout << "Session successfully created.\n";

    /*--------------------Load image------------------*/
    cout << "\n<-------------------Loading image------------------->" << endl;
    Mat img = imread(image_path);
    if(img.empty()) {
        cout << "Can't open the image!" << endl;
        return -1;
    }
    cvtColor(img, img, CV_BGR2RGB);

    Tensor resized_tensor(DT_FLOAT, TensorShape({1, input_height, input_width, 3}));

    CVMat_to_Tensor(img, &resized_tensor, input_height, input_width);

    cout << resized_tensor.DebugString() << endl;

    cout << "\n<---------Running the model with test_image-------->" << endl;
//    auto input_tensor = resized_tensor.scalar<float>();
    vector<Tensor> outputs;
    Status status_run = session->Run({{input_tensor_name, resized_tensor}}, {output_nodes}, {}, &outputs);

    if(!status_run.ok()) {
        cout << "ERROR: run failed..." << endl;
        cout << status_run.ToString() << endl;
        return 1;
    }

    cout << "Output tensor size:" << outputs.size() << endl;
    for(const auto &output:outputs) {
        cout << output.DebugString() << endl;
    }

    Tensor result = outputs[0];
    result.SubSlice(0);
//    auto result_map = result.tensor<long long, 3>();
//    cout << result_map << endl;


//    tensor2Mat(result, result_map, input_height, input_width);

    /*-----------------Tensor to Mat------------------------*/
    float *p = result.flat<float>().data();
    Mat mask(513, 513, CV_8UC1);
    for(int i = 0; i < input_height; ++i) {
        for(int j = 0; j < input_width; ++j) {
            mask.at<uint8>(i, j) = argmax(p, p + 21);
            p += 21;
        }
    }

    cout << "\n<----------------LUT---------------------->" << endl;
    Mat label_colors = imread(label_colors_path, 1);
    Mat mask_color = mask.clone();
    Mat mask_final;
//    cvtColor(mask_color, mask_color, CV_GRAY2BGR);
    LUT(mask_color, label_colors, mask_final);
    imshow("mask", mask);
    imshow("mask_color", mask_color);
    imshow("mask_final", mask_final);
    cvWaitKey(0);
    return 0;
}
