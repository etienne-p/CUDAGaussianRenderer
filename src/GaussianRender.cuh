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


__device__ __host__ struct CameraData {
  glm::mat4 viewProjection;
  glm::mat4 projection;
  glm::mat4 view;
  glm::vec2 fovCotangent;
  glm::vec2 depthScaleBias;
  glm::vec3 position;
  float aspect;
};

__device__ __host__ struct Rect {
  glm::vec2 min;
  glm::vec2 max;

  __device__ __host__ glm::vec2 GetCenter() const {
    return min + (max - min) * 0.5f;
  }
};

__device__ __host__ struct Ellipse {
  glm::vec2 center;
  glm::vec2 extent;
  glm::vec2 cosSin;

  __device__ __host__ glm::vec2 GetPrincipalAxis() const {
    return cosSin * extent.x;
  }
  __device__ __host__ glm::vec2 GetMinorAxis() const {
    return glm::vec2(cosSin.y, -cosSin.x) * extent.y;
  }
};

// Updated at most once per frame.
__device__ __host__ struct GlobalArgs {
  CameraData cameraData;
  float4* positions;
  float4* scales;
  float4* rotations;
  float4* colors;
  int sphericalHarmonicsDegree;
  int sphericalHarmonicsCount;
  float* sphericalHarmonics;
  float2* positionsClipSpaceXY;
  float* positionsClipSpaceZ;
  float4* conics;
  float4* screenEllipses;
  int32_t primitiveCount;
  uchar4* backBuffer;
  int32_t* tileRanges;
};

__device__ __host__ struct TileListArgs {
  uint64_t* keys;
  int32_t* values;
  int32_t size;
  int32_t capacity;
};

template <typename T>
struct DoubleBuffer {
 private:
  T* buffers_[2];
  int selector_;

 public:
  DoubleBuffer(T* current, T* alternate) {
    selector_ = 0;
    buffers_[0] = current;
    buffers_[1] = alternate;
  }

  T* Current() const { return buffers_[selector_]; }

  T* Alternate() const { return buffers_[selector_ ^ 1]; }

  int Selector() const { return selector_; }
};

void SetGlobalArgs(GlobalArgs* global_args);
void SetTileListArgs(TileListArgs* tile_list_args);

void EvaluateSphericalHarmonics(CudaTimer& timer, int32_t count);

void EvaluateSplatClipData(CudaTimer& timer, int32_t count);

int32_t BuildTileList(CudaTimer& timer, int32_t num_blocks,
                      int32_t tile_list_capacity);

void SortTileList(CudaTimer& timer, int32_t tile_list_size,
                  void*& device_temp_storage,
                  size_t& temp_storage_size_in_bytes,
                  DoubleBuffer<uint64_t>& keys, DoubleBuffer<int32_t>& values);

void EvaluateTileRange(CudaTimer& timer, int32_t tile_list_size);

void RasterizeTile(CudaTimer& timer);
