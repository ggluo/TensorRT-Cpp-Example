echo "Running all tests"

echo "========export the onnx model=========="
python data/export_model.py data/model.onnx

echo "==========compile the tensorRT inference code=========="
make 

echo "==========run the tensorRT inference code from C++ =========="
./main data/model.onnx data/first_engine.trt

echo "==========run the tensorRT inference code from C =========="
./main_c data/model.onnx data/first_engine.trt