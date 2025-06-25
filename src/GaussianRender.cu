#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "CudaErrorCheck.cpp"
#include "GaussianRender.cuh"

// Try out Epanechnikov kernel.
// #define EPANECHNIKOV_KERNEL

// Load via the read-only data cache.
// Applicable to global data that will only be read by the kernel.
template <typename T>
__device__ T loadReadOnly(T* ptr)
{
    return __ldg(reinterpret_cast<const T*>(ptr));
}

// Conversion between builtin CUDA types and glm:: types.
__device__ __host__ __forceinline__ glm::vec2 builtinToGlmVec2(const float2 v)
{
    return glm::vec2(v.x, v.y);
}

__device__ __host__ __forceinline__ glm::vec4 builtinToGlmVec4(const float4 v)
{
    return glm::vec4(v.x, v.y, v.z, v.w);
}

__device__ __host__ __forceinline__ glm::ivec4 builtinToGlmVec4i(const int4 v)
{
    return glm::ivec4(v.x, v.y, v.z, v.w);
}

// Global counters.
// Used to track processed splats when building the tiles list.
__device__ __managed__ int32_t g_SplatCounter;
// Used to track the size of the tiles list.
__device__ __managed__ int32_t g_TileCounter;

// Structs holding pointers to global memory and related sizes & capacities.
__constant__ GlobalArgs g_GlobalArgs;
__constant__ TileListArgs g_TileListArgs;

void setGlobalArgs(GlobalArgs* globalArgs)
{
    checkCudaErrors(cudaMemcpyToSymbol(g_GlobalArgs, globalArgs, sizeof(GlobalArgs), 0, cudaMemcpyHostToDevice));
}

void setTileListArgs(TileListArgs* tileListArgs)
{
    checkCudaErrors(cudaMemcpyToSymbol(g_TileListArgs, tileListArgs, sizeof(TileListArgs), 0, cudaMemcpyHostToDevice));
}

// Infer clip space data from world space data for each splat.
__global__ void evaluateSplatClipDataKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < g_GlobalArgs.splatCount)
    {
        auto splatPositionData = loadReadOnly(&g_GlobalArgs.positionWorldSpace[index]);
        // TODO: Should we layout all first vectors then all second vectors? (~ SoA vs AoS.)
        auto splatCovarianceData0 = loadReadOnly(&g_GlobalArgs.covarianceWorldSpace[index * 2 + 0]);
        auto splatCovarianceData1 = loadReadOnly(&g_GlobalArgs.covarianceWorldSpace[index * 2 + 1]);

        auto splatPosition = glm::vec3(splatPositionData.x, splatPositionData.y, splatPositionData.z);

        // Unpack covariance.
        auto splatCovariance = glm::mat3();
        splatCovariance[0][0] = splatCovarianceData0.x;
        splatCovariance[1][0] = splatCovarianceData0.y;
        splatCovariance[2][0] = splatCovarianceData0.z;
        splatCovariance[1][1] = splatCovarianceData1.x;
        splatCovariance[2][1] = splatCovarianceData1.y;
        splatCovariance[2][2] = splatCovarianceData1.z;
        // Copy symmetric part.
        splatCovariance[0][1] = splatCovariance[1][0];
        splatCovariance[0][2] = splatCovariance[2][0];
        splatCovariance[1][2] = splatCovariance[2][1];

        // Centroid of the gaussian in view space.
        auto viewPosition = (glm::vec3)(g_GlobalArgs.cameraData.view * glm::vec4(splatPosition, 1));
        // We precompute camera related variables such as cotangents and Z parameters.
        auto fovCotangent = g_GlobalArgs.cameraData.fovCotangent;
        // We map the Z axis linearly, similarly to an orthographic projection.
        // This gives us better depth precision for sorting tiles later.
        auto depthScaleBias = g_GlobalArgs.cameraData.depthScaleBias;

        // Para-perspective, affine approximation of the projection matrix.
        // Includes a basis change (identity with scale X and Y negated).
        auto zRcp = 1.0f / viewPosition.z;
        auto zRcpSqr = zRcp * zRcp;
        auto scaleX = -fovCotangent.x * zRcp;
        auto scaleY = -fovCotangent.y * zRcp;
        auto fovCotangentTimesTrs = fovCotangent * viewPosition.xy;
        auto shearX = fovCotangentTimesTrs.x * zRcpSqr;
        auto shearY = fovCotangentTimesTrs.y * zRcpSqr;
        auto translationX = -fovCotangentTimesTrs.x * zRcp;
        auto translationY = -fovCotangentTimesTrs.y * zRcp;

        // Build para-perspective matrix.
        // It corresponds to the Jacobian detailed in "Object Space EWA Surface Splatting".
        auto affineProjection = glm::mat4x4(1);
        // Scale.
        affineProjection[0][0] = scaleX;
        affineProjection[1][1] = scaleY;
        affineProjection[2][2] = depthScaleBias.x;
        // Shear.
        affineProjection[2][0] = shearX;
        affineProjection[2][1] = shearY;
        // Translation.
        affineProjection[3][0] = translationX;
        affineProjection[3][1] = translationY;
        affineProjection[3][2] = depthScaleBias.y;

        // Note that translation is dropped.
        auto viewProjection = (glm::mat3) affineProjection * (glm::mat3) g_GlobalArgs.cameraData.view;
        // Evaluate covariance in clip space.
        auto clipCovariance = viewProjection * glm::transpose(splatCovariance) * glm::transpose(viewProjection);
        auto clipPosition = (glm::vec3)(affineProjection * glm::vec4(viewPosition, 1));

        // Evaluate the eigen decomposition of the 2D covariance matrix to obtain
        // an oriented bounding rectangle for the splat.
        // TODO: Drop the splat if determinant is ~zero?
        auto det = clipCovariance[0][0] * clipCovariance[1][1] - clipCovariance[1][0] * clipCovariance[1][0];
        // Trace over two.
        auto mid = 0.5f * (clipCovariance[0][0] + clipCovariance[1][1]);
        // Compute eigen values, see
        // https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html.
        constexpr float epsilon = 1e-12f;
        auto lambda0 = mid + glm::sqrt(glm::max(epsilon, mid * mid - det));
        auto lambda1 = mid - glm::sqrt(glm::max(epsilon, mid * mid - det));

        auto eigenVector0 = glm::normalize(glm::vec2(clipCovariance[1][0], lambda0 - clipCovariance[0][0]));
        // Note that we use camera aspect to straighten the rectangle, aka obtain perpendicular vectors.
        auto eigenVector1 = glm::normalize(glm::vec2(eigenVector0.y / g_GlobalArgs.cameraData.aspect, -eigenVector0.x));

        // From eigenvectors to bounding ellipse.
        auto angle = glm::atan(eigenVector0.y, eigenVector0.x);
        // Multiplication by 3: 3 Sigmas rule.
        // Note: There is a generally linear relationship between overall performance and the sigma factor.
#ifdef EPANECHNIKOV_KERNEL
        auto extent = glm::sqrt(glm::vec2(lambda0, lambda1)) * glm::sqrt(7.0f);
#else
        auto extent = glm::sqrt(glm::vec2(lambda0, lambda1)) * 3.0f;
#endif
        // Inverse of 2D covariance: the conic matrix.
        // Note: if the determinant is zero, the inverse is not defined.
        auto invDet = 1.0f / glm::max(epsilon, det);
        // Only need 3 values since the matrix is symmetric.
        auto conic = glm::vec3(clipCovariance[1][1], -clipCovariance[1][0], clipCovariance[0][0]) * invDet;

        // Global writes.
        g_GlobalArgs.positionClipSpaceXY[index] = float2{clipPosition.x, clipPosition.y};
        g_GlobalArgs.positionClipSpaceZ[index] = clipPosition.z;
        g_GlobalArgs.screenEllipse[index] = float4{glm::cos(angle), glm::sin(angle), extent.x, extent.y};
        g_GlobalArgs.conic[index] = float4{conic.x, conic.y, conic.z, 0};
    }
}

void evaluateSplatClipData(CudaTimer& timer, int32_t count)
{
    // Each block processes 256 splats.
    // Dispatch as many groups as required.
    constexpr int32_t threadPerBlock = 256;
    const int32_t numBlocks = (count + threadPerBlock - 1) / threadPerBlock;
    const auto dimGrid = dim3(threadPerBlock);
    const auto dimBlock = dim3(numBlocks);

    timer.start();
    evaluateSplatClipDataKernel<<<dimBlock, dimGrid>>>();
    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateSplatsClipData\n");
    }
}

// Convert a point to ellipse local coordinates.
__device__ __forceinline__ glm::vec2 convertToEllipseCoordinates(const Ellipse& ellipse, const glm::vec2& point)
{
    // First handle translation.
    auto delta = point - ellipse.center;
    // Then rotation and scale.
    return glm::vec2((delta.x * ellipse.cosSin.x + delta.y * ellipse.cosSin.y) / ellipse.extent.x,
                     (delta.y * ellipse.cosSin.x - delta.x * ellipse.cosSin.y) / ellipse.extent.y);
}

// Test the intersection between a segment {v0, v1} and the unit circle.
__device__ __forceinline__ bool intersectsUnitCircle(const glm::vec2& v0, const glm::vec2& v1)
{
    auto delta = v1 - v0;
    auto lengthSqr = glm::length2(delta);
    // Find the projection of the center (0, 0) on the segment (v0, v1).
    // We clamp t in [0,1] to constrain the projection on the segment.
    auto t = __saturatef(glm::dot(-v0, delta) / lengthSqr);
    // Closest point to the unit circle center on the segment.
    auto projection = v0 + t * (v1 - v0);
    // Square is fine since we compare to one.
    return glm::length2(projection) < 1.0f;
}

// Test the intersection between an ellipse and a rectangle.
__device__ __forceinline__ bool ellipseRectOverlap(const Ellipse& ellipse, const Rect& rect)
{
    // Either the ellipse contains the rectangle, the rectangle contains the ellipse, or they
    // intersect. No early returns, would not be beneficial at warp scale.
    // (Would introduce divergence within warps.)

    // Test whether the ellipse center is contained within the rectangle.
    auto overlaps = //
        ellipse.center.x > rect.min.x && //
        ellipse.center.x < rect.max.x && //
        ellipse.center.y > rect.min.y && //
        ellipse.center.y < rect.max.y; //

    // Test whether the rectangle center is contained within the ellipse.
    overlaps |= glm::length2(convertToEllipseCoordinates(ellipse, rect.getCenter())) < 1.0f;

    // Convert rectangle vertices to local ellipse coordinates.
    glm::vec2 points[4];
    points[0] = convertToEllipseCoordinates(ellipse, rect.min);
    points[1] = convertToEllipseCoordinates(ellipse, glm::vec2(rect.max.x, rect.min.y));
    points[2] = convertToEllipseCoordinates(ellipse, rect.max);
    points[3] = convertToEllipseCoordinates(ellipse, glm::vec2(rect.min.x, rect.max.y));

    // Test whether rectangle edges intersect with the ellipse.
    overlaps |= intersectsUnitCircle(points[0], points[1]);
    overlaps |= intersectsUnitCircle(points[1], points[2]);
    overlaps |= intersectsUnitCircle(points[2], points[3]);
    overlaps |= intersectsUnitCircle(points[3], points[0]);

    return overlaps;
}

// Evaluates the axis aligned bounding box of an ellipse.
__device__ __forceinline__ Rect getAABBRect(const Ellipse& ellipse)
{
    auto right = ellipse.getPrincipalAxis();
    auto up = ellipse.getMinorAxis();

    // Initialize to extreme values so that they will be overriden.
    Rect rect;
    rect.min = glm::vec2(1e12f, 1e12f);
    rect.max = glm::vec2(-1e12f, -1e12f);

#pragma unroll
    for (auto j = 0; j != 4; ++j)
    {
        // Infer corner coordinates from bit twiddling.
        auto bottomBit = j & 1;
        auto topBit = j >> 1;
        // Produces [-1, 1, 1, -1].
        auto r = (bottomBit ^ topBit) * 2.0f - 1.0f;
        // Produces [-1, -1, 1, 1].
        auto u = (topBit) * 2.0f - 1.0f;
        auto v = right * r + up * u;
        rect.min = glm::min(rect.min, v);
        rect.max = glm::max(rect.max, v);
    }

    rect.min += ellipse.center;
    rect.max += ellipse.center;
    return rect;
}

constexpr uint32_t k_WarpMask = 0xffffffff;
constexpr int32_t k_WarpSize = 32;
constexpr int32_t k_WarpHalfSize = k_WarpSize / 2;
constexpr int32_t k_BuildTileListsThreadsPerGroup = k_WarpSize * 8;
constexpr uint32_t k_MaxUint32 = std::numeric_limits<uint32_t>::max();

// Combine tile index and clip depth to form a sorting key.
__device__ __forceinline__ uint64_t getKey(int32_t tileIndex, float clipDepth)
{
    // Recall that we have hardcoded the final buffer size, 1024x1024.
    // 1024 / 16 = 64 -> 4096 tiles total, 12bits.
    // Our projection linearly maps [near, far] to [0, 1].
    // We use the full 32 bits for maximum precision.
    auto quantizedDepth = (uint32_t) (glm::clamp(clipDepth) * k_MaxUint32);
    return ((uint64_t) tileIndex << 32) | quantizedDepth;
}

__global__ void buildTileListKernel()
{
    // Track splats chunks.
    __shared__ int32_t s_SplatStartIndex;
    __shared__ int32_t s_SplatCount;

    // Capacity matches one warp, the first warp which is responsible for loading splats.
    __shared__ int4 s_TilesRects[k_WarpSize];
    __shared__ float4 s_ScreenEllipse[k_WarpSize];
    __shared__ float2 s_PositionClipSpaceXY[k_WarpSize];
    __shared__ float s_Depths[k_WarpSize];
    __shared__ int s_TileExclusiveScan[k_WarpSize];

    // Capacity matches the number of threads in the group.
    __shared__ int s_ExpandedTiles[k_BuildTileListsThreadsPerGroup];
    __shared__ int s_ExpandedTileCount;
    __shared__ int s_TotalTilesCount;
    __shared__ int s_CumulatedExpandedTileCount;
    __shared__ bool s_HasPendingTiles;

    __shared__ int64_t s_TileKeys[k_BuildTileListsThreadsPerGroup];
    __shared__ int32_t s_TileValues[k_BuildTileListsThreadsPerGroup];
    __shared__ int32_t s_TileIndex;
    __shared__ int32_t s_WriteTileStartIndex;
    __shared__ int32_t s_WriteTileEndIndex;

    // Initialize shared counters.
    if (threadIdx.x == 0)
    {
        s_TileIndex = 0;
        s_HasPendingTiles = false;
    }

    __syncthreads();

    // Enter persistent loop.
    for (;;)
    {
        // The first thread pulls some splats.
        // We process chunks of splats to reduce contention on the global counter g_GaussianCounter.
        if (threadIdx.x == 0)
        {
            auto splatCounter = atomicAdd(&g_SplatCounter, k_WarpSize);
            s_SplatStartIndex = glm::min(splatCounter, g_GlobalArgs.splatCount);
            auto splatEndIndex = glm::min(splatCounter + k_WarpSize, g_GlobalArgs.splatCount);
            s_SplatCount = splatEndIndex - s_SplatStartIndex;
            s_ExpandedTileCount = 0;
            s_CumulatedExpandedTileCount = 0;
        }

        __syncthreads();

        // There are no splats left to process, exit.
        if (s_SplatCount == 0)
        {
            return;
        }

        // Makes sure any lingering data in the expanded scan is ignored.
        auto tileCount = 0;

        // Read splats data, coalesced.
        if (threadIdx.x < s_SplatCount)
        {
            // Read current splat data. The one this thread is responsible for.
            auto srcIndex = s_SplatStartIndex + threadIdx.x;
            auto positionClipSpaceXYData = loadReadOnly(&g_GlobalArgs.positionClipSpaceXY[srcIndex]);
            auto ellipseData = loadReadOnly(&g_GlobalArgs.screenEllipse[srcIndex]);

            auto ellipse = Ellipse();
            ellipse.center = glm::vec2(positionClipSpaceXYData.x, positionClipSpaceXYData.y);
            ellipse.cosSin = glm::vec2(ellipseData.x, ellipseData.y);
            ellipse.extent = glm::vec2(ellipseData.z, ellipseData.w);

            auto rect = getAABBRect(ellipse);

            // From clip to tiles.
            auto tilesRectFloat = (glm::vec4(rect.min, rect.max) + 1.0f) * 0.5f * (float) k_TilesPerScreen;

            // Clip within screen bounds.
            auto tilesRect = glm::ivec4(
                glm::max(0, (int) glm::floor(tilesRectFloat.x)),
                glm::max(0, (int) glm::floor(tilesRectFloat.y)),
                glm::min(k_TilesPerScreen, (int) glm::ceil(tilesRectFloat.z)),
                glm::min(k_TilesPerScreen, (int) glm::ceil(tilesRectFloat.w)));

            // Min-Max to Center-Size representation.
            tilesRect.zw -= tilesRect.xy;

            // We must account for negative surface of the rectangle.
            // Note: We could prevent this earlier. But it would require a discard mechnism earlier in the pipeline.
            tileCount = glm::max(0, tilesRect.z * tilesRect.w);

            // Store shared data.
            s_Depths[threadIdx.x] = loadReadOnly(&g_GlobalArgs.positionClipSpaceZ[srcIndex]);
            s_TilesRects[threadIdx.x] = int4{tilesRect.x, tilesRect.y, tilesRect.z, tilesRect.w};
            s_ScreenEllipse[threadIdx.x] = ellipseData;
            s_PositionClipSpaceXY[threadIdx.x] = positionClipSpaceXYData;
        }

        // __syncwarp() is sufficient since only the first warp is working right now.
        __syncwarp();

        // Evaluate exclusive scan.
        // It is done outside of the previous condition, since we may have less splats than threads in the warp.
        if (threadIdx.x < k_WarpSize)
        {
            // Only the first warp passes here, so threadIdx.x is the lane.
            auto inclusiveScan = tileCount;
#pragma unroll
            for (auto delta = 1u; delta <= k_WarpHalfSize; delta <<= 1u)
            {
                auto n = __shfl_up_sync(k_WarpMask, inclusiveScan, delta);

                // Only process the threads that do have some work.
                // We never clear the shared array, don't touch what lingers from previous runs.
                // Note: first warp -> threadIdx == laneId.
                if (threadIdx.x >= delta)
                {
                    inclusiveScan += n;
                }
            }

            s_TileExclusiveScan[threadIdx.x] = inclusiveScan - tileCount;

            if (threadIdx.x == k_WarpSize - 1)
            {
                s_TotalTilesCount = inclusiveScan;
            }

            // Each thread writes its tiles in shared memory, to be processed later by the whole group.
            // Exclusive scan allows each thread to evaluate its writing indices in shared memory.
            auto startWriteIndex = glm::min(inclusiveScan - tileCount, k_BuildTileListsThreadsPerGroup);
            auto endWriteIndex = glm::min(inclusiveScan, k_BuildTileListsThreadsPerGroup);
            auto writeIndex = startWriteIndex;

            // Write tiles in shared memory.
            while (__ballot_sync(k_WarpMask, writeIndex < endWriteIndex) != 0)
            {
                if (writeIndex < endWriteIndex)
                {
                    s_ExpandedTiles[writeIndex] = threadIdx.x;
                    ++writeIndex;
                }
            }

            __syncwarp();

            // The last thread writing to shared memory is responsible for the tracking of shared counters.
            if (endWriteIndex - startWriteIndex > 0)
            {
                // If everything fits there's no remaining work.
                // All tiles fit in one pass.
                if (endWriteIndex == s_TotalTilesCount)
                {
                    s_HasPendingTiles = false;
                    s_ExpandedTileCount = endWriteIndex;
                    s_CumulatedExpandedTileCount = s_ExpandedTileCount;
                }
                // There is remaining work.
                // We will need multiple passes.
                else if (endWriteIndex == k_BuildTileListsThreadsPerGroup)
                {
                    s_HasPendingTiles = true;
                    s_ExpandedTileCount = endWriteIndex;
                    s_CumulatedExpandedTileCount = s_ExpandedTileCount;
                }
            }
        }

        // Wait for shared memory updates to be visible to the whole group.
        __syncthreads();

        for (;;)
        {
            // Overlap test between a tile and an ellipse corresponding to the outline of a splat.
            if (threadIdx.x < s_ExpandedTileCount)
            {
                auto splatIndex = s_ExpandedTiles[threadIdx.x];
                auto tilesRect = builtinToGlmVec4i(s_TilesRects[splatIndex]);
                auto localTileIndex =
                    threadIdx.x - s_TileExclusiveScan[splatIndex] + s_CumulatedExpandedTileCount - s_ExpandedTileCount;

                // Coordinates of the current tile within the splat AABB rect.
                auto localTileCoords = glm::ivec2(localTileIndex % tilesRect.z, localTileIndex / tilesRect.z);
                // Coordinates of the current tile within the screen.
                auto globalTileCoords = tilesRect.xy + localTileCoords;
                constexpr float tileNormalizedSize = k_TileSize / (float) k_ScreenSize;
                constexpr float tileClipSize = tileNormalizedSize * 2.0f;
                // Coordinates of the current tile in clip space.
                auto tileClip = (glm::vec2)(globalTileCoords) *tileClipSize - 1.0f;
                // Note: We assume an aspect ratio of 1.
                // Tile rect in clip space.
                Rect tileRectClipSpace;
                tileRectClipSpace.min = tileClip;
                tileRectClipSpace.max = tileClip + glm::vec2(tileClipSize, tileClipSize);

                auto ellipse = Ellipse();
                ellipse.center = glm::vec2(s_PositionClipSpaceXY[splatIndex].x, s_PositionClipSpaceXY[splatIndex].y);
                ellipse.cosSin = glm::vec2(s_ScreenEllipse[splatIndex].x, s_ScreenEllipse[splatIndex].y);
                ellipse.extent = glm::vec2(s_ScreenEllipse[splatIndex].z, s_ScreenEllipse[splatIndex].w);

                if (ellipseRectOverlap(ellipse, tileRectClipSpace))
                {
                    // The insertion index must be valid since there's room for every thread in the group.
                    auto localInsertionIndex = atomicAdd(&s_TileIndex, 1);
                    auto globalTileIndex = globalTileCoords.y * k_TilesPerScreen + globalTileCoords.x;
                    s_TileKeys[localInsertionIndex] = getKey(globalTileIndex, s_Depths[splatIndex]);
                    s_TileValues[localInsertionIndex] = s_SplatStartIndex + splatIndex;
                }
            }

            __syncthreads();

            // Commit tiles to global memory if needed.
            if (s_TileIndex > 0)
            {
                // The first thread increments the global counter.
                if (threadIdx.x == 0)
                {
                    auto tileTileListCounter = atomicAdd(&g_TileCounter, s_TileIndex);
                    // Restrict to target capacity.
                    s_WriteTileStartIndex = glm::min(tileTileListCounter, g_TileListArgs.capacity);
                    s_WriteTileEndIndex = glm::min(tileTileListCounter + s_TileIndex, g_TileListArgs.capacity);
                    // Reset.
                    s_TileIndex = 0;
                }

                __syncthreads();

                // Abort and exit if global memory is full.
                if (s_WriteTileEndIndex == g_TileListArgs.capacity)
                {
                    return;
                }

                // Write global memory.
                if (threadIdx.x < s_WriteTileEndIndex - s_WriteTileStartIndex)
                {
                    auto dstIndex = s_WriteTileStartIndex + threadIdx.x;
                    g_TileListArgs.keys[dstIndex] = s_TileKeys[threadIdx.x];
                    g_TileListArgs.values[dstIndex] = s_TileValues[threadIdx.x];
                }
            }

            __syncthreads();

            // Break if there's no tiles left to process for the current set of splats.
            // We will attempt to load and process a new set of splats.
            if (!s_HasPendingTiles)
            {
                break;
            }

            // The first warp is responsible for distributing a new set of tiles in shared memory.
            if (threadIdx.x < k_WarpSize)
            {
                // Evaluate the tiles pending processing for the current splat.
                auto exclusiveScan = s_TileExclusiveScan[threadIdx.x] - s_CumulatedExpandedTileCount;
                auto startWriteIndex = glm::clamp(exclusiveScan, 0, k_BuildTileListsThreadsPerGroup);
                auto endWriteIndex = glm::clamp(exclusiveScan + tileCount, 0, k_BuildTileListsThreadsPerGroup);
                auto writeIndex = startWriteIndex;

                // Write the next batch of tiles to shared memory.
                while (__ballot_sync(k_WarpMask, writeIndex < endWriteIndex) != 0)
                {
                    if (writeIndex < endWriteIndex)
                    {
                        s_ExpandedTiles[writeIndex] = threadIdx.x;
                        ++writeIndex;
                    }
                }

                __syncwarp();

                // The last thread writing to shared memory is responsible for the tracking of shared counters.
                if (endWriteIndex - startWriteIndex > 0)
                {
                    // If everything fits there's no remaining work.
                    // This will be the last pass.
                    if (endWriteIndex == s_TotalTilesCount - s_CumulatedExpandedTileCount)
                    {
                        s_HasPendingTiles = false;
                        s_ExpandedTileCount = endWriteIndex;
                        s_CumulatedExpandedTileCount += s_ExpandedTileCount;
                    }
                    // There is remaining work.
                    // We still have more passes left.
                    else if (endWriteIndex == k_BuildTileListsThreadsPerGroup)
                    {
                        s_HasPendingTiles = true;
                        s_ExpandedTileCount = endWriteIndex;
                        s_CumulatedExpandedTileCount += s_ExpandedTileCount;
                    }
                }
            }

            __syncthreads();
        }
    }
}

// A helper used to set and get the number of tiles emitted on the device.
static int32_t initCounter = 0;

// Emit tiles per splat.
int32_t buildTileList(CudaTimer& timer, int32_t numBlocks, int32_t tileListCapacity)
{
    // Reset counters. (Splats and tile render list.)
    initCounter = 0;
    checkCudaErrors(cudaMemcpyToSymbol(g_SplatCounter, &initCounter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(g_TileCounter, &initCounter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));

    // Note that number of groups is passed by the caller.
    // Recall that we have persistent group: they run as long as there is remaining work.
    // This is why the workload is not considered for dispatch parameters.
    const auto dimGrid = dim3(k_BuildTileListsThreadsPerGroup);
    const auto dimBlock = dim3(numBlocks);

    timer.start();
    buildTileListKernel<<<dimBlock, dimGrid>>>();
    timer.stop();

    checkCudaErrors(cudaMemcpyFromSymbol(&initCounter, g_TileCounter, sizeof(int32_t), 0, cudaMemcpyDeviceToHost));

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error buildTileList\n");
    }

    // Minimum since atomicAdd(...) has no rollback in case of excess.
    return glm::min(initCounter, tileListCapacity);
}

// Sort the list of tiles.
// Done with cub::, which requires a temporary buffer that will be reallocated as needed.
void sortTileList(CudaTimer& timer,
                  int32_t tileListSize,
                  void*& deviceTempStorage,
                  size_t& tempStorageSizeInBytes,
                  DoubleBuffer<uint64_t>& keys,
                  DoubleBuffer<int32_t>& values)
{
    timer.start();

    // We can skip the top bits in the key.
    // 32 bits for depth, 12 bits for tile index.
    // (Depends on the number of tiles on screen, which is hardcoded.)
    constexpr int32_t beginBit = 0;
    constexpr int32_t endBit = 32 + 12;

    // Instantiate cub:: double buffers.
    cub::DoubleBuffer<uint64_t> cubKeys(keys.current(), keys.alternate());
    cub::DoubleBuffer<int32_t> cubValues(values.current(), values.alternate());

    // Dry run. Allocate temp memory if needed. From the documentation about "d_temp_storage":
    // "When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done."
    size_t requiredTempStorageSizeInBytes;
    cub::DeviceRadixSort::SortPairs(
        (void*) nullptr, requiredTempStorageSizeInBytes, cubKeys, cubValues, tileListSize, beginBit, endBit);

    if (requiredTempStorageSizeInBytes > tempStorageSizeInBytes)
    {
        if (deviceTempStorage != nullptr)
        {
            checkCudaErrors(cudaFree(deviceTempStorage));
        }
        checkCudaErrors(cudaMalloc((void**) &deviceTempStorage, requiredTempStorageSizeInBytes));
        tempStorageSizeInBytes = requiredTempStorageSizeInBytes;
    }

    // Actual sorting.
    cub::DeviceRadixSort::SortPairs(
        deviceTempStorage, tempStorageSizeInBytes, cubKeys, cubValues, tileListSize, beginBit, endBit);

    // Copy double buffers back to host, so that we know which buffer to use from now on.
    keys = DoubleBuffer<uint64_t>(cubKeys.Current(), cubKeys.Alternate());
    values = DoubleBuffer<int32_t>(cubValues.Current(), cubValues.Alternate());

    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("error sortTileList\n");
    }
}

__global__ void evaluateTileRangesKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    // The first thread within the grid handles first and last elements.
    // Not that bad, only one warp within the grid will have divergence.
    if (index == 0)
    {
        auto firstKey = loadReadOnly(&g_TileListArgs.keys[0]);
        auto firstTileIndex = firstKey >> 32;
        g_GlobalArgs.tileRange[firstTileIndex * 2] = 0;

        auto lastKey = loadReadOnly(&g_TileListArgs.keys[g_TileListArgs.size - 1]);
        auto lastTileIndex = lastKey >> 32;
        g_GlobalArgs.tileRange[lastTileIndex * 2 + 1] = g_TileListArgs.size;
    }
    else if (index < g_TileListArgs.size)
    {
        // Pairwise comparison, look at the previous element.
        auto prevKey = loadReadOnly(&g_TileListArgs.keys[index - 1]);
        auto prevTileIndex = prevKey >> 32;
        auto key = loadReadOnly(&g_TileListArgs.keys[index]);
        auto tileIndex = key >> 32;

        // If we detect a tile change,
        // update the end of the previous range and the start of the current one.
        if (tileIndex != prevTileIndex)
        {
            g_GlobalArgs.tileRange[prevTileIndex * 2 + 1] = index;
            g_GlobalArgs.tileRange[tileIndex * 2] = index;
        }
    }
}

// Evaluate ranges within the list corresponding to each tile.
void evaluateTileRange(CudaTimer& timer, int32_t tileListSize)
{
    constexpr int32_t threadPerBlock = 256;
    const int32_t numBlocks = (tileListSize + threadPerBlock - 1) / threadPerBlock;
    const auto dimGrid = dim3(threadPerBlock);
    const auto dimBlock = dim3(numBlocks);

    timer.start();
    evaluateTileRangesKernel<<<dimBlock, dimGrid>>>();
    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateTileRanges\n");
    }
}

__global__ void rasterizeTilesKernel()
{
    __shared__ int32_t s_FirstSplatIndex;
    __shared__ int32_t s_SplatsCount;
    __shared__ float4 s_Colors[k_WarpSize];
    __shared__ float4 s_Conics[k_WarpSize];
    __shared__ float2 s_Centers[k_WarpSize];

    // The first thread pulls the range beginning and end.
    if (threadIdx.x == 0)
    {
        // Block Id corresponds to the tile.
        s_FirstSplatIndex = loadReadOnly(&g_GlobalArgs.tileRange[blockIdx.x * 2]);
        auto endSplatIndex = loadReadOnly(&g_GlobalArgs.tileRange[blockIdx.x * 2 + 1]);
        s_SplatsCount = endSplatIndex - s_FirstSplatIndex;
    }

    __syncthreads();

    // If the range is empty, exit.
    if (s_SplatsCount == 0)
    {
        return;
    }

    // Coordinates of the pixel handled by this thread within the tile .
    const auto threadTilePixelCoords = glm::ivec2(threadIdx.x % k_TileSize, threadIdx.x / k_TileSize);
    // Coordinates of the pixel within the assembled image.
    const auto threadBufferPixelCoords =
        glm::ivec2(blockIdx.x % k_TilesPerScreen, blockIdx.x / k_TilesPerScreen) * k_TileSize + threadTilePixelCoords;
    // TODO: Account for pixel center offset (0.5, 0.5)?
    const auto clipCoords = (glm::vec2) threadBufferPixelCoords * (2.0f / (float) k_ScreenSize) - 1.0f;

    // Initialize color and transmittance for blending. (Volumetric Rendering.)
    auto color = glm::vec3(0.0f);
    auto transmittance = 1.0f;

    while (s_SplatsCount > 0)
    {
        // Don't overflow shared buffers.
        // We only load one 32 splats chunk at once, in hopes of needing no more.
        // These loads aren't coalesced.
        auto splatsCount = glm::min(s_SplatsCount, k_WarpSize);

        __syncthreads();

        // Load splat data.
        if (threadIdx.x < splatsCount)
        {
            auto srcIndex = g_TileListArgs.values[s_FirstSplatIndex + threadIdx.x];
            s_Colors[threadIdx.x] = loadReadOnly(&g_GlobalArgs.color[srcIndex]);
            s_Conics[threadIdx.x] = loadReadOnly(&g_GlobalArgs.conic[srcIndex]);
            s_Centers[threadIdx.x] = loadReadOnly(&g_GlobalArgs.positionClipSpaceXY[srcIndex]);
        }

        // Wait for loaded data to be visible to all threads.
        __syncthreads();

        // Rasterize the current chunk.
        for (auto i = 0; i != splatsCount; ++i)
        {
            // Square distance to splat center.
            auto d = clipCoords - builtinToGlmVec2(s_Centers[i]);
            auto splatColor = builtinToGlmVec4(s_Colors[i]);
            auto conic = s_Conics[i];
            // (T(d) * conic * d)
            auto dx = conic.x * d.x * d.x + conic.z * d.y * d.y + 2.0f * conic.y * d.x * d.y;
#ifdef EPANECHNIKOV_KERNEL
            auto density = 1.0f - dx / 7.0f;
#else
            auto density = __expf(-0.5f * dx);
#endif
            auto alpha = splatColor.a * __saturatef(density);
            color += splatColor.rgb * transmittance * alpha;
            transmittance *= (1.0f - alpha);
        }

        // The first thread updates the shared counters.
        if (threadIdx.x == 0)
        {
            s_FirstSplatIndex += splatsCount;
            s_SplatsCount -= splatsCount;
        }

        // If the tile is opaque, stop rasterization.
        if (__syncthreads_count(transmittance > 0.02f) == 0)
        {
            break;
        }
    }

    uchar4 quantizedColor;
    quantizedColor.x = color.x * 255;
    quantizedColor.y = color.y * 255;
    quantizedColor.z = color.z * 255;
    quantizedColor.w = 255;

    // Commit to global memory.
    auto globalWriteIndex = threadBufferPixelCoords.y * k_ScreenSize + threadBufferPixelCoords.x;
    g_GlobalArgs.backBuffer[globalWriteIndex] = quantizedColor;
}

// Rasterize tiles.
void rasterizeTile(CudaTimer& timer)
{
    // We dispatch one block per tile, one thread per pixel within the tile.
    constexpr int32_t threadPerBlock = k_TileSize * k_TileSize;
    const auto dimGrid = dim3(threadPerBlock);
    const auto dimBlock = dim3(k_TotalTiles);

    timer.start();
    rasterizeTilesKernel<<<dimBlock, dimGrid>>>();
    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error renderDepthBuffer\n");
    }
}
