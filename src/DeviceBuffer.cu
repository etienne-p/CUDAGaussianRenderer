#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "CudaErrorCheck.cpp"

template <typename T>
class DeviceBuffer
{
  private:
    T* m_Ptr{nullptr};
    std::size_t m_Size{0};
    std::size_t m_AllocatedSize{0};
    void releaseIfNeeded();

  public:
    std::size_t size() const
    {
        return m_Size;
    };
    T* getPtr() const
    {
        return m_Ptr;
    };
    DeviceBuffer()
        : m_Ptr(nullptr)
        , m_Size(0)
        , m_AllocatedSize(0) {};
    DeviceBuffer(std::size_t size, bool shouldClearMemory = false);
    DeviceBuffer(std::vector<T> data);
    ~DeviceBuffer();
    bool resizeIfNeeded(std::size_t size);
    void copyFrom(const std::vector<T>& data);
    void copyTo(std::vector<T>& data);
    void copyTo(std::vector<T>& data, std::size_t count);
    void clearMemory(int32_t value);
};

template <typename T>
void DeviceBuffer<T>::clearMemory(int32_t value)
{
    checkCudaErrors(cudaMemset(m_Ptr, value, m_Size * sizeof(T)));
}

template <typename T>
bool DeviceBuffer<T>::resizeIfNeeded(std::size_t size)
{
    if (size != m_Size)
    {
        // Reallocate if we need more space.
        if (size > m_AllocatedSize)
        {
            releaseIfNeeded();
            m_Size = size;
            m_AllocatedSize = size;
            checkCudaErrors(cudaMalloc((void**) &m_Ptr, m_AllocatedSize * sizeof(T)));
            return true;
        }
        // Just update used space.
        m_Size = size;
    }
    return false;
}

template <typename T>
void DeviceBuffer<T>::releaseIfNeeded()
{
    if (m_Ptr != nullptr)
    {
        checkCudaErrors(cudaFree(m_Ptr));
        m_Ptr = nullptr;
    }
    m_Size = 0;
    m_AllocatedSize = 0;
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::size_t size, bool shouldClearMemory)
{
    resizeIfNeeded(size);
    if (shouldClearMemory)
    {
        clearMemory(0);
    }
};

template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::vector<T> data)
{
    copyFrom(data);
};

template <typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
    releaseIfNeeded();
};

template <typename T>
void DeviceBuffer<T>::copyFrom(const std::vector<T>& data)
{
    resizeIfNeeded(data.size());
    checkCudaErrors(cudaMemcpy(m_Ptr, &data[0], data.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void DeviceBuffer<T>::copyTo(std::vector<T>& data, std::size_t count)
{
    assert(count <= m_Size);
    if (data.size() != count)
    {
        data.resize(count);
    }
    checkCudaErrors(cudaMemcpy(&data[0], m_Ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void DeviceBuffer<T>::copyTo(std::vector<T>& data)
{
    copyTo(data, data.size());
}
