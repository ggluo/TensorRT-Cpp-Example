#ifndef UTILS_H
#define UTILS_H
#include "NvInfer.h"
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <cassert>
#include <numeric>

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using trt_unique_ptr = std::unique_ptr<T, InferDeleter>;

struct infer_params
{
    std::string weights_file; // currently supports only uff file
    int batch_size;
    std::string save_engine;
    std::string load_engine;
};

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        default: throw std::runtime_error("Invalid DataType.");
    }
}

using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger
{
public:
    explicit Logger(Severity severity = Severity::kWARNING)
        : mReportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) noexcept override
    {
        std::cout << "[TRT] " << std::string(msg) << std::endl;
    }


    void setReportableSeverity(Severity severity) noexcept
    {
        mReportableSeverity = severity;
    }
    Severity getReportableSeverity() const
    {
        return mReportableSeverity;
    }

private:
    
    static const char* severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }


    static std::ostream& severityOstream(Severity severity)
    {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    Severity mReportableSeverity;
};


static auto StreamDeleter = [](cudaStream_t* pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(0);                                                                         \
        }                                                                                      \
    }
#endif

inline void* safeCudaMalloc(size_t memSize) {
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline void safeCudaFree(void* deviceMem) {
    CUDA_CHECK(cudaFree(deviceMem));
}

inline bool load_engine(){

}

inline bool save_engine(const nvinfer1::ICudaEngine& engine, std::string const& fileName, std::ostream& err)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        err << "Cannot open engine file: " << fileName << std::endl;
        return false;
    }

    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine{engine.serialize()};
    if (serializedEngine == nullptr)
    {
        err << "Engine serialization failed" << std::endl;
        return false;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

#endif