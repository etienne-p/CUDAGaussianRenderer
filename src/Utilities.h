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
    GLuint m_BufferId;
    std::size_t m_Size;

  public:
    struct VertexArrayBinding
    {
      private:
        GLuint m_Index;

      public:
        VertexArrayBinding(GLuint bufferId, GLuint index)
        {
            m_Index = index;
            glEnableVertexAttribArray(index);
            glBindBuffer(target, bufferId);
        }
        ~VertexArrayBinding()
        {
            glDisableVertexAttribArray(m_Index);
        }
    };

    struct Binding
    {
      public:
        Binding(GLuint bufferId)
        {
            glBindBuffer(target, bufferId);
        }
        ~Binding()
        {
            glBindBuffer(target, 0);
        }
    };

    GLBuffer(std::size_t size, const T* data, GLenum usage)
    {
        m_Size = size;

        glGenBuffers(1, &m_BufferId);
        glBindBuffer(target, m_BufferId);
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
        glDeleteBuffers(1, &m_BufferId);
    }

    GLuint getBufferId() const
    {
        return m_BufferId;
    }
    std::size_t getSize() const
    {
        return m_Size;
    }
    std::size_t getSizeInBytes() const
    {
        return m_Size * sizeof(T);
    }
    VertexArrayBinding getBinding(GLuint index)
    {
        return VertexArrayBinding(m_BufferId, index);
    }
    Binding getBinding()
    {
        return Binding(m_BufferId);
    }
};

struct GLVertexArray
{
  private:
    GLuint m_Id;

  public:
    GLVertexArray()
    {
        glGenVertexArrays(1, &m_Id);
        glBindVertexArray(m_Id);
    }
    ~GLVertexArray()
    {
        glDeleteVertexArrays(1, &m_Id);
    }
};

struct CudaGraphicsResource
{
  private:
    cudaGraphicsResource* m_Resource;

  public:
    template <typename T>
    struct Binding
    {
      private:
        T* m_Ptr;
        cudaGraphicsResource* m_Resource;

      public:
        Binding(cudaGraphicsResource* resource)
        {
            m_Resource = resource;
            std::size_t size; // Unused.
            checkCudaErrors(cudaGraphicsMapResources(1, &resource, 0));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**) &m_Ptr, &size, resource));
        }
        ~Binding()
        {
            cudaGraphicsUnmapResources(1, &m_Resource, 0);
        }
        T* getPtr() const
        {
            return m_Ptr;
        }
    };
    CudaGraphicsResource(GLuint bufferId, uint32_t flags)
    {
        cudaGraphicsGLRegisterBuffer(&m_Resource, bufferId, flags);
    }
    ~CudaGraphicsResource()
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_Resource));
    }
    template <typename T>
    Binding<T> getBinding()
    {
        return Binding<T>(m_Resource);
    }
};

struct CudaTimer
{
  private:
    cudaEvent_t m_Start;
    cudaEvent_t m_Stop;
    float m_ElapsedTimeMs{0};

  public:
    CudaTimer()
    {
        cudaEventCreate(&m_Start);
        cudaEventCreate(&m_Stop);
    }
    ~CudaTimer()
    {
        cudaEventDestroy(m_Start);
        cudaEventDestroy(m_Stop);
    }
    void start()
    {
        cudaEventRecord(m_Start);
    }
    void stop()
    {
        cudaEventRecord(m_Stop);
        cudaEventSynchronize(m_Stop);
        cudaEventElapsedTime(&m_ElapsedTimeMs, m_Start, m_Stop);
    }
    float getElapseTimedMs() const
    {
        return m_ElapsedTimeMs;
    }
};
