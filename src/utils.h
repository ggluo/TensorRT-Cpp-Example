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
#include <vector>
#include <complex>
#include <sstream>
#include <iterator>
#include <limits>

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
    infer_params(const std::string& wfile, int bsize, const std::string& efile, const std::string& tfile)
        : weights_file(wfile), batch_size(bsize), save_engine(efile), load_engine(tfile) {}
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
    //explicit Logger(Severity severity = Severity::kWARNING)
      //  : reportableSeverity(severity)
    //{}
    Logger(Severity severity = Severity::kWARNING)
    {
        const char* logLevel = std::getenv("TENSORRT_LOG_LEVEL");
        if (logLevel)
        {
            std::string level(logLevel);
            if (level == "INTERNAL_ERROR")
                reportableSeverity = Severity::kINTERNAL_ERROR;
            else if (level == "ERROR")
                reportableSeverity = Severity::kERROR;
            else if (level == "WARNING")
                reportableSeverity = Severity::kWARNING;
            else if (level == "INFO")
                reportableSeverity = Severity::kINFO;
            else if (level == "VERBOSE")
                reportableSeverity = Severity::kVERBOSE;
            else
                reportableSeverity = Severity::kWARNING; // default level
        }
        else
        {
            reportableSeverity = severity;
        }
    }

     void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= reportableSeverity)
        {
            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cout << "INFO: "; break;
                case Severity::kVERBOSE: std::cout << "VERBOSE: "; break;
                default: std::cout << "UNKNOWN: "; break;
            }
            std::cout << msg << std::endl;
        }
    }


    void setReportableSeverity(Severity severity) noexcept
    {
        reportableSeverity = severity;
    }
    Severity getReportableSeverity() const
    {
        return reportableSeverity;
    }

private:

    Severity reportableSeverity;
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

bool load_engine(const std::string& fileName, trt_unique_ptr<nvinfer1::ICudaEngine>& m_engine,
 trt_unique_ptr<nvinfer1::IRuntime>&m_runtime, trt_unique_ptr<nvinfer1::IExecutionContext>& m_context, std::unique_ptr<Logger>& m_logger);

bool save_engine(const nvinfer1::ICudaEngine& engine, std::string const& fileName, std::ostream& err);

void write_cfl(const std::string& filename, const std::vector<std::complex<float>>& array, const std::vector<int>& dims);

void read_cfl(const std::string& filename, std::vector<std::complex<float>>& array, std::vector<int>& dims);

#endif