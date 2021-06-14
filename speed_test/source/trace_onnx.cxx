#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main(int argc, const char **argv)
{
    if(argc != 2) {
        cerr << "usage: " << argv[0]
             << " <path-to-exported-script-module>" << endl;
        return -1;
    }

    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(argv[1]);
    } catch(const cv::Exception &e) {
        cout << "onnx file load error" << endl;
        return -1;
    }


    // First execution takes super long, so just ignore it

    auto t1 = std::chrono::high_resolution_clock::now();
    // Second and next exucutions are now fast. Measure how fast.
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    cout << duration << endl;
    return 0;
}
