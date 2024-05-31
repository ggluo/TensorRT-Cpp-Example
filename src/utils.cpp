#include "utils.h"

bool load_engine(const std::string& fileName, trt_unique_ptr<nvinfer1::ICudaEngine>& m_engine,
trt_unique_ptr<nvinfer1::IRuntime>&m_runtime, trt_unique_ptr<nvinfer1::IExecutionContext>& m_context, 
std::unique_ptr<Logger>& m_logger)
{
    std::ifstream engineFile(fileName, std::ifstream::binary);

    assert(engineFile.is_open() && "Failed to open engine file");
    if (!engineFile)
    {
        std::cerr << "Cannot open engine file: " << fileName << std::endl;
        return false;
    }

    auto const start_pos = engineFile.tellg();
    engineFile.ignore(std::numeric_limits<std::streamsize>::max());
    size_t filesize = engineFile.gcount();
    engineFile.seekg(start_pos);
    std::unique_ptr<char[]> engineBuf(new char[filesize]);
    engineFile.read(engineBuf.get(), filesize);

    m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));
    assert(m_runtime != nullptr && "Failed to create runtime");

    m_engine.reset(m_runtime->deserializeCudaEngine((void*)engineBuf.get(), filesize));
    assert(m_engine != nullptr && "Failed to create engine");

    m_context.reset(m_engine->createExecutionContext());
    assert(m_context != nullptr && "Failed to create context");
    
    return true;
}

bool save_engine(const nvinfer1::ICudaEngine& engine, std::string const& fileName, std::ostream& err)
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

void write_cfl(const std::string& filename, const std::vector<std::complex<float>>& array, const std::vector<int>& dims) {
    std::ofstream hdrFile(filename + ".hdr");
    if (!hdrFile) {
        throw std::runtime_error("Cannot open file: " + filename + ".hdr");
    }
    hdrFile << "# Dimensions\n";
    for (const auto& dim : dims) {
        hdrFile << dim << " ";
    }
    hdrFile << "\n";
    hdrFile.close();

    std::ofstream cflFile(filename + ".cfl", std::ios::binary);
    if (!cflFile) {
        throw std::runtime_error("Cannot open file: " + filename + ".cfl");
    }
    cflFile.write(reinterpret_cast<const char*>(array.data()), array.size() * sizeof(std::complex<float>));
    cflFile.close();
}

std::vector<int> getDimsFromFile(const std::string& filename) {
    std::ifstream hdrFile(filename + ".hdr");
    if (!hdrFile) {
        throw std::runtime_error("Cannot open file: " + filename + ".hdr");
    }
    std::string line;
    std::getline(hdrFile, line);  // skip first line
    std::getline(hdrFile, line);  // read second line
    hdrFile.close();

    std::istringstream iss(line);
    std::vector<int> dims((std::istream_iterator<int>(iss)), std::istream_iterator<int>());
    return dims;
}

std::vector<std::complex<float>> read_cfl(const std::string& filename, std::vector<int>& dims) {
    dims = getDimsFromFile(filename);
    int totalElements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    
    std::ifstream cflFile(filename + ".cfl", std::ios::binary);
    if (!cflFile) {
        throw std::runtime_error("Cannot open file: " + filename + ".cfl");
    }

    std::vector<std::complex<float>> data(totalElements);
    cflFile.read(reinterpret_cast<char*>(data.data()), totalElements * sizeof(std::complex<float>));
    cflFile.close();

    return data;
}
