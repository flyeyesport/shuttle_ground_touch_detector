#include <torch/script.h>
#include <torch/cuda.h>
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

    torch::Device device = torch::kCPU;
    if(torch::cuda::is_available()) {
        device = torch::kCUDA;
    } else {
        std::cout << "CUDA is not available! Inference on GPU impossible." << std::endl;
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        module.to(device);
    } catch(const c10::Error &e) {
        cerr << "error loading the model: " << e.what() << endl;
        return -1;
    }

    // Create a vector of inputs.
    vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 1, 16, 128, 128}).to(device));

    // First execution takes super long, so just ignore it
    at::Tensor output = module.forward(inputs).toTensor();

    auto t1 = std::chrono::high_resolution_clock::now();
    // Second and next exucutions are now fast. Measure how fast.
    output = module.forward(inputs).toTensor();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    cout << duration << endl;
    return 0;
}
