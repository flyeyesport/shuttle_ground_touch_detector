#!/bin/bash

python source/convert_model_for_jit_c++_usage.py -m input/checkpoint.pth -o output/model.pt

#we get warnings when generating onnx model. The model uses some extra logic in if statements. I didn't find a way to solve it. I think that generated model is broken, but we can test its speed.
#python source/convert_model_for_onnx_c++_usage.py -o output/broken_model.onnx

rm -R build
mkdir build
cd build
cmake ../source
make

echo -e -n "001\tsqueezenet3d\t\tscript jit\t\tprobably good\t" >> ../../result.txt
./squeeze_net_c++_jit_inference ../output/model.pt >> ../../result.txt

# remove newline from last line of the result.txt file that was added by custom_cnn_rnn_3_c++_jit_inference
truncate -s-1 ../../result.txt
echo -e "\t\tModified SqueezeNet to 3D, with added shuttle positions, operates on sets of 8 consequtive frames with only middle 4 taken into account" >> ../../result.txt

#echo -e -n "001\tsqueezenet3d\t\ttrace onnx\t\tprobably broken\t" >> ../../result.txt
#./squeeze_net_c++_onnx_inference ../output/broken_model-opt-shape.onnx >> ../../result.txt


cd ..
rm -R build
