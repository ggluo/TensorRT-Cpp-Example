
#include "trt.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <cstdio>


bool trt_infer::build()
{
    if (m_params.load_engine.empty())
    {
        m_builder = trt_unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_logger));
        if (!m_builder)
        {
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = trt_unique_ptr<nvinfer1::INetworkDefinition>(m_builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            return false;
        }

        auto config = trt_unique_ptr<nvinfer1::IBuilderConfig>(m_builder->createBuilderConfig());
        if (!config)
        {
            return false;
        }

        auto parser = trt_unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *m_logger)); //TODO: support other formats
        if (!parser)
        {
            return false;
        }

        auto loaded = parser->parseFromFile(m_params.weights_file.c_str(), static_cast<int>(m_logger->getReportableSeverity()));
        if (!loaded)
        {
            return false;
        }

        // CUDA stream used for profiling by the builder.
        auto profileStream = makeCudaStream();
        if (!profileStream)
        {
            return false;
        }
        config->setProfileStream(*profileStream);

        trt_unique_ptr<nvinfer1::IHostMemory> plan{m_builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            return false;
        }

        m_runtime = trt_unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger));
        if (!m_runtime)
        {
            return false;
        }

        m_engine = trt_unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!m_engine)
        {
            return false;
        }
    }
    else
    {
        std::cout << "Loading engine from file: " << m_params.load_engine << std::endl;
        load_engine(m_params.load_engine, m_engine, m_runtime, m_context, m_logger);
    }

    if (!m_params.save_engine.empty())
    {
        std::cout << "Saving engine to file: " << m_params.save_engine << std::endl;
        save_engine(*m_engine, m_params.save_engine, std::cerr);
    }

    m_context = trt_unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context)
    {
        return false;
    }
    create_device_io_buffer();
    m_builder.reset(nullptr);
    return true;
}

void trt_infer::create_device_io_buffer()
{
    if (m_params.batch_size < 1)
    {
        throw std::invalid_argument("Batch size must be at least 1");
    }

    const auto total_bindings = m_engine->getNbBindings();

    m_bindings.resize(total_bindings);
    m_binding_sizes.resize(total_bindings);
    m_binding_names.resize(total_bindings);
    m_binding_dims.resize(total_bindings);
    m_binding_types.resize(total_bindings);

    for (auto i = 0; i < total_bindings; ++i)
    {
        const char* binding_name = m_engine->getBindingName(i);\
        nvinfer1::DataType binding_type = m_engine->getBindingDataType(i);
        nvinfer1::Dims binding_dims;

        if (dynamic_shapes) {
            std::cout << "Dynamics shapes is not supported!" << std::endl;
            exit(1);
        }
        else
        {
            binding_dims = m_engine->getBindingDimensions(i);
        }

        int64_t total_size = volume(binding_dims) * getElementSize(binding_type);
        m_binding_sizes[i]   = total_size;
        m_binding_names[i] = binding_name;
        m_binding_dims[i]  = binding_dims;
        m_binding_types[i] = binding_type;

        m_bindings[i] = safeCudaMalloc(total_size);
        if(m_engine->bindingIsInput(i))
            nr_input_bindings++;
        else
            nr_output_bindings++;
        
        #ifdef DEBUG
        if(m_engine->bindingIsInput(i)){
            std::cout << "Input binding -> " << binding_name << " -> ";
        }
        else{
            std::cout << "Output binding -> " << binding_name << " -> ";
        }
        printf("Index: %d, Name: %s, Type: %d, Dims: ", i, binding_name, binding_type);
        for (int j = 0; j < binding_dims.nbDims; ++j)
        {
            printf("%d ", binding_dims.d[j]);
        }
        std::cout << std::endl;
        #endif
    }
}

//TODO: check the type of the input, is it should be consistent with the type of the input binding
//TODO: check the size of the input, is it should be consistent with the size of the input binding
void trt_infer::CopyFromHostToDevice(const std::vector<float>& input,
                           int bind_index, const cudaStream_t& stream)
{
    #ifdef DEBUG
        printf("input size: %ld, binding size: %ld.\n", input.size()*sizeof(float), m_binding_sizes[bind_index]);
    #endif
    const auto inputSize = m_binding_sizes[bind_index];
    CUDA_CHECK(cudaMemcpyAsync(m_bindings[bind_index], input.data(), inputSize, cudaMemcpyHostToDevice, stream));
}

void trt_infer::CopyFromDeviceToHost(std::vector<float>& output, int bind_index,
                           const cudaStream_t& stream)
{
    #ifdef DEBUG
        printf("output size: %ld, binding size: %ld.\n", output.size()*sizeof(float), m_binding_sizes[bind_index]);
    #endif
    assert(output.size()*sizeof(float) == m_binding_sizes[bind_index]);
    const auto outputSize = m_binding_sizes[bind_index];
    CUDA_CHECK(cudaMemcpyAsync(output.data(), m_bindings[bind_index], outputSize, cudaMemcpyDeviceToHost, stream));
}

bool trt_infer::infer()
{
    return m_context->executeV2(&m_bindings[0]);
}

bool trt_infer::infer(const cudaStream_t& stream)
{
    return m_context->enqueueV2(&m_bindings[0], stream, nullptr);
}

trt_infer::~trt_infer()
{
for(size_t i=0;i<m_bindings.size();i++) {
        safeCudaFree(m_bindings[i]);
    }
}