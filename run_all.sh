echo "Running all tests"

echo "========export the onnx model=========="
python data/export_model.py data/model.onnx

echo "==========compile the tensorRT inference code=========="
make 

echo "==========run the tensorRT inference code=========="
./main data/model.onnx data/first_engine.trt
