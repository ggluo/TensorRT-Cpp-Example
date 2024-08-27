echo -e "*********Running all tests********\n"
echo -e "==============TEST1===================="
echo -e "========export the onnx model==========\n"
python python/export_model.py data/model.onnx
echo -e "\n"

echo -e "=========================TEST2========================="
echo -e "==========compile the tensorRT inference code==========\n"
make 
echo -e "\n"

echo -e "============================TEST3============================"
echo -e "==========run the tensorRT inference code from C++ ==========\n"
./main data/model.onnx data/first_engine.trt
echo -e "\n"

echo -e "==============TEST4===================="
echo -e "==========run the tensorRT inference from existing engile file ==========\n"
./main data/first_engine.trt
echo -e "\n"

echo -e "========================TEST5=============================="
echo -e "==========run the tensorRT inference code from C ==========\n"
./main_c data/model.onnx data/first_engine.trt
echo -e "\n"

echo -e "======================TEST6========================="
echo -e "=========run GaussianBlur test for pytorch==========\n"
python python/gaussian_blur_pytorch.py data/grd pytorch_blurred data/blur.onnx
./gaussian_blur data/blur.onnx data/blur_engine.trt data/grd data/output
python python/utils.py data/output tensorRT_blurred_pytorch
echo -e "\n"

echo -e "====================TEST7=============================="
echo -e "=========run GaussianBlur test for tensorflow==========\n"
python python/gaussian_blur_tf.py data/grd tf_blurred data/blur.onnx
./gaussian_blur data/blur.onnx data/blur_engine.trt data/grd data/output
python python/utils.py data/output tensorRT_blurred_tf