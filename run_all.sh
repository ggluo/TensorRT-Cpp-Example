echo "Running all tests"

echo "========export the onnx model=========="
python python/export_model.py data/model.onnx

echo "==========compile the tensorRT inference code=========="
make 

echo "==========run the tensorRT inference code from C++ =========="
./main data/model.onnx data/first_engine.trt

echo "==========use the existing engine file to run the tensorRT inference code from C++ =========="
./main data/first_engine.trt

echo "==========run the tensorRT inference code from C =========="
./main_c data/model.onnx data/first_engine.trt

echo "=========run GaussianBlur test=========="
python python/gaussian_blur_pytorch.py data/grd pytorch_blurred data/blur.onnx
./gaussian_blur data/blur.onnx data/blur_engine.trt data/grd data/output
python python/utils.py data/output tensorRT_blurred