#pragma once

#define GLM_FORCE_CUDA
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <cuda_runtime_api.h>
#include <cstdint>
#include "Consts.h"
#include "DeviceBuffer.cu"
#include "Utilities.h"
#include <assert.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

__device__ __host__ struct CameraData
{
    glm::mat4 viewProjection;
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec2 fovCotangent;
    glm::vec2 depthScaleBias;
    glm::vec3 position;
    float aspect;
};

__device__ __host__ struct Rect
{
    glm::vec2 min;
    glm::vec2 max;

    __device__ __host__ glm::vec2 getCenter() const
    {
        return min + (max - min) * 0.5f;
    }
};

__device__ __host__ struct Ellipse
{
    glm::vec2 center;
    glm::vec2 extent;
    glm::vec2 cosSin;

    __device__ __host__ glm::vec2 getPrincipalAxis() const
    {
        return cosSin * extent.x;
    }
    __device__ __host__ glm::vec2 getMinorAxis() const
    {
        return glm::vec2(cosSin.y, -cosSin.x) * extent.y;
    }
};

// Updated at most once per frame.
__device__ __host__ struct GlobalArgs
{
    CameraData cameraData;
    float4* position;
    float4* scaleAndRotation;
    float4* color;
    int sphericalHarmonicsDegree;
    int sphericalHarmonicsCount;
    float* sphericalHarmonics;
    // Xy and Z stored separately as they are consumed by different kernels.
    float2* positionClipSpaceXY;
    float* positionClipSpaceZ;
    // Pack conic and color as both are consumed by the rasterization kernel.
    float4* conic;
    float4* screenEllipse;
    int32_t splatCount;
    uchar4* backBuffer;
    int32_t* tileRange;
};

// Distinct from GlobalArgs since it needs multiple updates over the frame.
__device__ __host__ struct TileListArgs
{
    uint64_t* keys;
    int32_t* values;
    int32_t size;
    int32_t capacity;
};

// Similar to cub::DoubleBuffer<T>,
// introduced so that client code doesn't need to reference cub:: types.
template <typename T>
struct DoubleBuffer
{
  private:
    T* m_Buffers[2];
    int m_Selector;

  public:
    DoubleBuffer(T* current, T* alternate)
    {
        m_Selector = 0;
        m_Buffers[0] = current;
        m_Buffers[1] = alternate;
    }

    T* current() const
    {
        return m_Buffers[m_Selector];
    }

    T* alternate() const
    {
        return m_Buffers[m_Selector ^ 1];
    }

    int selector() const
    {
        return m_Selector;
    }
};

// Set constant structs holding pointers to global memory and related sizes & capacities.
void setGlobalArgs(GlobalArgs* globalArgs);
void setTileListArgs(TileListArgs* tileListArgs);

// Evaluate view dependent spherical harmonics (degree 1+).
void evaluateSphericalHarmonics(CudaTimer& timer, int32_t count);

// Per splat initial data transformation.
void evaluateSplatClipData(CudaTimer& timer, int32_t count);

// Emit tiles per splat.
int32_t buildTileList(CudaTimer& timer, int32_t numBlocks, int32_t tileListCapacity);

// Sort the list of tiles.
void sortTileList(CudaTimer& timer,
                  int32_t tileListSize,
                  void*& deviceTempStorage,
                  size_t& tempStorageSizeInBytes,
                  DoubleBuffer<uint64_t>& keys,
                  DoubleBuffer<int32_t>& values);

// Evaluate ranges within the list corresponding to each tile.
void evaluateTileRange(CudaTimer& timer, int32_t tileListSize);

// Rasterize tiles.
void rasterizeTile(CudaTimer& timer);
