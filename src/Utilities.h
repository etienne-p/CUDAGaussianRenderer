#pragma once

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <vector>
#include "CudaErrorCheck.cpp"

template <typename T, GLenum target>
struct GLBuffer
{
  private:
    GLuint buffer_id_;
    std::size_t size_;

  public:
    struct VertexArrayBinding
    {
      private:
        GLuint index_;

      public:
        VertexArrayBinding(GLuint buffer_id, GLuint index)
        {
            index_ = index;
            glEnableVertexAttribArray(index);
            glBindBuffer(target, buffer_id);
        }
        ~VertexArrayBinding()
        {
            glDisableVertexAttribArray(index_);
        }
    };

    struct Binding
    {
      public:
        Binding(GLuint buffer_id)
        {
            glBindBuffer(target, buffer_id);
        }
        ~Binding()
        {
            glBindBuffer(target, 0);
        }
    };

    GLBuffer(std::size_t size, const T* data, GLenum usage)
    {
        size_ = size;

        glGenBuffers(1, &buffer_id_);
        glBindBuffer(target, buffer_id_);
        glBufferData(target, sizeof(T) * size, data, usage);
    }
    GLBuffer(std::size_t size, const T* data)
        : GLBuffer(size, data, GL_STATIC_DRAW)
    {}
    GLBuffer(const std::vector<T>& data)
        : GLBuffer(data.size(), &data[0], GL_STATIC_DRAW)
    {}
    GLBuffer(std::size_t size)
        : GLBuffer(size, nullptr, GL_DYNAMIC_DRAW_ARB)
    {}
    ~GLBuffer()
    {
        glDeleteBuffers(1, &buffer_id_);
    }

    GLuint GetBufferId() const
    {
        return buffer_id_;
    }
    std::size_t GetSize() const
    {
        return size_;
    }
    std::size_t GetSizeInBytes() const
    {
        return size_ * sizeof(T);
    }
    VertexArrayBinding GetBinding(GLuint index)
    {
        return VertexArrayBinding(buffer_id_, index);
    }
    Binding GetBinding()
    {
        return Binding(buffer_id_);
    }
};

struct GLVertexArray
{
  private:
    GLuint id_;

  public:
    GLVertexArray()
    {
        glGenVertexArrays(1, &id_);
        glBindVertexArray(id_);
    }
    ~GLVertexArray()
    {
        glDeleteVertexArrays(1, &id_);
    }
};

struct CudaGraphicsResource
{
  private:
    cudaGraphicsResource* resource_;

  public:
    template <typename T>
    struct Binding
    {
      private:
        T* mapped_ptr_;
        cudaGraphicsResource* cuda_resource_;

      public:
        Binding(cudaGraphicsResource* resource)
        {
            cuda_resource_ = resource;
            std::size_t size; // Unused.
            CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &resource, 0));
            CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &mapped_ptr_, &size, resource));
        }
        ~Binding()
        {
            cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
        }
        T* GetPtr() const
        {
            return mapped_ptr_;
        }
    };
    CudaGraphicsResource(GLuint buffer_id, uint32_t flags)
    {
        cudaGraphicsGLRegisterBuffer(&resource_, buffer_id, flags);
    }
    ~CudaGraphicsResource()
    {
        CHECK_CUDA_ERRORS(cudaGraphicsUnregisterResource(resource_));
    }
    template <typename T>
    Binding<T> GetBinding()
    {
        return Binding<T>(resource_);
    }
};

struct CudaTimer
{
  private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    float elapsed_time_ms_{0};

  public:
    CudaTimer()
    {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }
    ~CudaTimer()
    {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }
    void Start()
    {
        cudaEventRecord(start_event_);
    }
    void Stop()
    {
        cudaEventRecord(stop_event_);
        cudaEventSynchronize(stop_event_);
        cudaEventElapsedTime(&elapsed_time_ms_, start_event_, stop_event_);
    }
    float GetElapsedTimeMs() const
    {
        return elapsed_time_ms_;
    }
};
