#ifndef TRT_H
#define TRT_H

#include "utils.h"
#include "NvInfer.h"

#include <string>
#include <memory>

class trt_infer
{
//TODO: to handle different input order NCHW, NHWC, etc
//TODO: to handle different model format, uff, onnx, etc
//TODO: to load engine from file
//TODO: to create profile optimization in builder
public:
    trt_infer(infer_params const& params)
    : m_params(params), m_runtime(nullptr), m_engine(nullptr)
{
    if (m_params.load_engine.empty() && m_params.weights_file.empty())
    {
        throw std::invalid_argument("Either load_engine or weights_file must be specified");
    }

    m_logger = std::make_unique<Logger>(Severity::kINFO);
    if (!m_logger)
    {
        throw std::runtime_error("Failed to create logger");
    }
}

    ~trt_infer();

    trt_infer(const trt_infer&) = delete; //disallows copy operations between its instances
    trt_infer& operator=(const trt_infer&) = delete;  //disallows assignment operations between its instances


    //! \brief build the TensorRT engine for a network
    bool build();

    //! \brief run the engine to perform inference
    bool infer();

    bool infer(const cudaStream_t& stream);

    void create_device_io_buffer();

    void CopyFromHostToDevice(const std::vector<float>& input,
                               int bindIndex, const cudaStream_t& stream);

    void CopyFromDeviceToHost(std::vector<float>& output, int bindIndex,
                               const cudaStream_t& stream);

protected:

    infer_params m_params;
    std::unique_ptr<Logger> m_logger{nullptr};

    trt_unique_ptr<nvinfer1::IBuilder> m_builder{nullptr}; // TensorRT builder used to create the engine

    trt_unique_ptr<nvinfer1::IRuntime> m_runtime{nullptr};  // TensorRT runtime used to deserialize the engine

    trt_unique_ptr<nvinfer1::ICudaEngine> m_engine{nullptr}; // TensorRT engine used to run the network

    trt_unique_ptr<nvinfer1::IExecutionContext> m_context{nullptr}; // TensorRT context used to perform inference

    nvinfer1::IOptimizationProfile* m_profile = nullptr;

    void CreateDeviceBuffer();

    std::vector<void*> m_bindings; // pointers to input and output data
    std::vector<size_t> m_binding_sizes; // sizes of input and output data
    std::vector<nvinfer1::Dims> m_binding_dims; // dimensions of input and output data
    std::vector<nvinfer1::DataType> m_binding_types; // data types of input and output data
    std::vector<std::string> m_binding_names; // names of input and output data

    int nr_input_bindings = 0;
    int nr_output_bindings = 0;

    bool dynamic_shapes = false;
};

#endif