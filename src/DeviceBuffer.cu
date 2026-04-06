#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "CudaErrorCheck.cpp"

template <typename T>
class DeviceBuffer
{
  private:
    T* ptr_{nullptr};
    std::size_t size_{0};
    std::size_t allocated_size_{0};
    void ReleaseIfNeeded();

  public:
    std::size_t Size() const
    {
        return size_;
    };
    T* GetPtr() const
    {
        return ptr_;
    };
    DeviceBuffer()
        : ptr_(nullptr)
        , size_(0)
        , allocated_size_(0) {};
    DeviceBuffer(std::size_t size, bool should_clear_memory = false);
    DeviceBuffer(std::vector<T> data);
    ~DeviceBuffer();
    bool ResizeIfNeeded(std::size_t size);
    void CopyFrom(const std::vector<T>& data);
    void CopyTo(std::vector<T>& data);
    void CopyTo(std::vector<T>& data, std::size_t count);
    void ClearMemory(int32_t value);
};

template <typename T>
void DeviceBuffer<T>::ClearMemory(int32_t value)
{
    CHECK_CUDA_ERRORS(cudaMemset(ptr_, value, size_ * sizeof(T)));
}

template <typename T>
bool DeviceBuffer<T>::ResizeIfNeeded(std::size_t size)
{
    if (size != size_)
    {
        if (size > allocated_size_)
        {
            ReleaseIfNeeded();
            size_ = size;
            allocated_size_ = size;
            CHECK_CUDA_ERRORS(cudaMalloc((void**) &ptr_, allocated_size_ * sizeof(T)));
            return true;
        }
        size_ = size;
    }
    return false;
}

template <typename T>
void DeviceBuffer<T>::ReleaseIfNeeded()
{
    if (ptr_ != nullptr)
    {
        CHECK_CUDA_ERRORS(cudaFree(ptr_));
        ptr_ = nullptr;
    }
    size_ = 0;
    allocated_size_ = 0;
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::size_t size, bool should_clear_memory)
{
    ResizeIfNeeded(size);
    if (should_clear_memory)
    {
        ClearMemory(0);
    }
};

template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::vector<T> data)
{
    CopyFrom(data);
};

template <typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
    ReleaseIfNeeded();
};

template <typename T>
void DeviceBuffer<T>::CopyFrom(const std::vector<T>& data)
{
    ResizeIfNeeded(data.size());
    if (data.size() != 0)
    {
        CHECK_CUDA_ERRORS(cudaMemcpy(ptr_, &data[0], data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template <typename T>
void DeviceBuffer<T>::CopyTo(std::vector<T>& data, std::size_t count)
{
    assert(count <= size_);
    if (data.size() != count)
    {
        data.resize(count);
    }
    CHECK_CUDA_ERRORS(cudaMemcpy(&data[0], ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void DeviceBuffer<T>::CopyTo(std::vector<T>& data)
{
    CopyTo(data, data.size());
}
