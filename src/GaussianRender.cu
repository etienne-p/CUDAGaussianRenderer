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
__device__ T LoadReadOnly(T* ptr)
{
    return __ldg(reinterpret_cast<const T*>(ptr));
}

// Conversion between builtin CUDA types and glm:: types.
__device__ __host__ __forceinline__ glm::vec2 BuiltinToGlmVec2(const float2 v)
{
    return glm::vec2(v.x, v.y);
}

__device__ __host__ __forceinline__ glm::vec4 BuiltinToGlmVec4(const float4 v)
{
    return glm::vec4(v.x, v.y, v.z, v.w);
}

__device__ __host__ __forceinline__ glm::ivec4 BuiltinToGlmVec4i(const int4 v)
{
    return glm::ivec4(v.x, v.y, v.z, v.w);
}

// Global counters.
// Used to track processed splats when building the tiles list.
__device__ __managed__ int32_t g_splat_counter;
// Used to track the size of the tiles list.
__device__ __managed__ int32_t g_tile_counter;

// Structs holding pointers to global memory and related sizes & capacities.
__constant__ GlobalArgs g_global_args;
__constant__ TileListArgs g_tile_list_args;

void SetGlobalArgs(GlobalArgs* globalArgs)
{
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(g_global_args, globalArgs, sizeof(GlobalArgs), 0, cudaMemcpyHostToDevice));
}

void SetTileListArgs(TileListArgs* tileListArgs)
{
    CHECK_CUDA_ERRORS(
        cudaMemcpyToSymbol(g_tile_list_args, tileListArgs, sizeof(TileListArgs), 0, cudaMemcpyHostToDevice));
}

__device__ glm::vec4 DecodeVec4(uint32_t v)
{
    return glm::vec4((v >> 24u) & 0xFF, (v >> 16u) & 0xFF, (v >> 8u) & 0xFF, v & 0xFF) / 255.0f;
}

// ! Code generated using the sh_gen.py python script. !
// clang-format off
__device__ glm::vec3 EvalSphericalHarmonics(const int l, const glm::vec3& dir, const float* sh, const int stride)
{
    const auto x = dir.x;
    const auto y = dir.y;
    const auto z = dir.z;
    auto result = glm::vec3(0.0f);

    // Level 0.
    auto sh0 = glm::vec3(LoadReadOnly(&sh[0 * stride]), LoadReadOnly(&sh[1 * stride]), LoadReadOnly(&sh[2 * stride]));

    result +=
        (0.282094792F) * sh0;

    // Level 1.
    if (l > 0)
    {
        auto sh1 = glm::vec3(LoadReadOnly(&sh[3 * stride]), LoadReadOnly(&sh[4 * stride]), LoadReadOnly(&sh[5 * stride]));
        auto sh2 = glm::vec3(LoadReadOnly(&sh[6 * stride]), LoadReadOnly(&sh[7 * stride]), LoadReadOnly(&sh[8 * stride]));
        auto sh3 = glm::vec3(LoadReadOnly(&sh[9 * stride]), LoadReadOnly(&sh[10 * stride]), LoadReadOnly(&sh[11 * stride]));

        result +=
            (0.488602512F*y) * sh1 +
            (0.488602512F*z) * sh2 +
            (0.488602512F*x) * sh3;

        // Level 2.
        if (l > 1)
        {
            auto xx = x * x;
            auto yy = y * y;
            auto zz = z * z;

            auto sh4 = glm::vec3(LoadReadOnly(&sh[12 * stride]), LoadReadOnly(&sh[13 * stride]), LoadReadOnly(&sh[14 * stride]));
            auto sh5 = glm::vec3(LoadReadOnly(&sh[15 * stride]), LoadReadOnly(&sh[16 * stride]), LoadReadOnly(&sh[17 * stride]));
            auto sh6 = glm::vec3(LoadReadOnly(&sh[18 * stride]), LoadReadOnly(&sh[19 * stride]), LoadReadOnly(&sh[20 * stride]));
            auto sh7 = glm::vec3(LoadReadOnly(&sh[21 * stride]), LoadReadOnly(&sh[22 * stride]), LoadReadOnly(&sh[23 * stride]));
            auto sh8 = glm::vec3(LoadReadOnly(&sh[24 * stride]), LoadReadOnly(&sh[25 * stride]), LoadReadOnly(&sh[26 * stride]));

            result +=
                (1.09254843F*x*y) * sh4 +
                (1.09254843F*y*z) * sh5 +
                (-0.946174696F*xx - 0.946174696F*yy + 0.630783131F) * sh6 +
                (1.09254843F*x*z) * sh7 +
                (0.546274215F*(x - y)*(x + y)) * sh8;

            // Level 3.
            if (l > 2)
            {
                auto sh9 = glm::vec3(LoadReadOnly(&sh[27 * stride]), LoadReadOnly(&sh[28 * stride]), LoadReadOnly(&sh[29 * stride]));
                auto sh10 = glm::vec3(LoadReadOnly(&sh[30 * stride]), LoadReadOnly(&sh[31 * stride]), LoadReadOnly(&sh[32 * stride]));
                auto sh11 = glm::vec3(LoadReadOnly(&sh[33 * stride]), LoadReadOnly(&sh[34 * stride]), LoadReadOnly(&sh[35 * stride]));
                auto sh12 = glm::vec3(LoadReadOnly(&sh[36 * stride]), LoadReadOnly(&sh[37 * stride]), LoadReadOnly(&sh[38 * stride]));
                auto sh13 = glm::vec3(LoadReadOnly(&sh[39 * stride]), LoadReadOnly(&sh[40 * stride]), LoadReadOnly(&sh[41 * stride]));
                auto sh14 = glm::vec3(LoadReadOnly(&sh[42 * stride]), LoadReadOnly(&sh[43 * stride]), LoadReadOnly(&sh[44 * stride]));
                auto sh15 = glm::vec3(LoadReadOnly(&sh[45 * stride]), LoadReadOnly(&sh[46 * stride]), LoadReadOnly(&sh[47 * stride]));

                result +=
                    (0.295021795F*y*(6.0F*xx - 2.0F*yy)) * sh9 +
                    (2.89061144F*x*y*z) * sh10 +
                    (3.6563664F*y*(-0.625F*xx - 0.625F*yy + 0.5F)) * sh11 +
                    (0.373176333F*z*(-5.0F*xx - 5.0F*yy + 2.0F)) * sh12 +
                    (0.457045799F*x*(-5.0F*xx - 5.0F*yy + 4.0F)) * sh13 +
                    (1.44530572F*z*(x - y)*(x + y)) * sh14 +
                    (0.59004359F*x*(xx - 3.0F*yy)) * sh15;

                // Level 4.
                if (l > 3)
                {
                    auto sh16 = glm::vec3(LoadReadOnly(&sh[48 * stride]), LoadReadOnly(&sh[49 * stride]), LoadReadOnly(&sh[50 * stride]));
                    auto sh17 = glm::vec3(LoadReadOnly(&sh[51 * stride]), LoadReadOnly(&sh[52 * stride]), LoadReadOnly(&sh[53 * stride]));
                    auto sh18 = glm::vec3(LoadReadOnly(&sh[54 * stride]), LoadReadOnly(&sh[55 * stride]), LoadReadOnly(&sh[56 * stride]));
                    auto sh19 = glm::vec3(LoadReadOnly(&sh[57 * stride]), LoadReadOnly(&sh[58 * stride]), LoadReadOnly(&sh[59 * stride]));
                    auto sh20 = glm::vec3(LoadReadOnly(&sh[60 * stride]), LoadReadOnly(&sh[61 * stride]), LoadReadOnly(&sh[62 * stride]));
                    auto sh21 = glm::vec3(LoadReadOnly(&sh[63 * stride]), LoadReadOnly(&sh[64 * stride]), LoadReadOnly(&sh[65 * stride]));
                    auto sh22 = glm::vec3(LoadReadOnly(&sh[66 * stride]), LoadReadOnly(&sh[67 * stride]), LoadReadOnly(&sh[68 * stride]));
                    auto sh23 = glm::vec3(LoadReadOnly(&sh[69 * stride]), LoadReadOnly(&sh[70 * stride]), LoadReadOnly(&sh[71 * stride]));
                    auto sh24 = glm::vec3(LoadReadOnly(&sh[72 * stride]), LoadReadOnly(&sh[73 * stride]), LoadReadOnly(&sh[74 * stride]));

                    result +=
                        (2.50334294F*x*y*(xx - yy)) * sh16 +
                        (0.295021795F*y*z*(18.0F*xx - 6.0F*yy)) * sh17 +
                        (1.26156626F*x*y*(-5.25F*xx - 5.25F*yy + 4.5F)) * sh18 +
                        (1.78412412F*y*z*(-2.625F*xx - 2.625F*yy + 1.5F)) * sh19 +
                        (7.40498828F*xx*yy - 4.23142188F*xx + 3.70249414F*xx*xx - 4.23142188F*yy + 3.70249414F*yy*yy + 0.846284375F) * sh20 +
                        (0.669046544F*x*z*(-7.0F*xx - 7.0F*yy + 4.0F)) * sh21 +
                        (-0.473087348F*(x - y)*(x + y)*(7.0F*xx + 7.0F*yy - 6.0F)) * sh22 +
                        (1.77013077F*x*z*(xx - 3.0F*yy)) * sh23 +
                        (-3.75501441F*xx*yy + 0.625835735F*xx*xx + 0.625835735F*yy*yy) * sh24;
                }
            }
        }
    }
    return glm::clamp(result + 0.5f);
}
// clang-format on

__global__ void EvaluateSphericalHarmonicsKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < g_global_args.primitiveCount)
    {
        auto positionData = LoadReadOnly(&g_global_args.positions[index]);

        auto splatPosition = glm::vec3(positionData.x, positionData.y, positionData.z);

        // Bundle opacity with position. Otherwise we'd have to read from color.
        auto opacity = positionData.w;

        // We align spherical harmonics using 4 floats for aligned access.
        auto rayDir = glm::normalize(g_global_args.cameraData.position - splatPosition);

        // Each spherical harmonic is laid out contiguously for the threads of the group.
        auto shIdx = blockIdx.x * blockDim.x * g_global_args.sphericalHarmonicsCount + threadIdx.x;

        // Note: stride is not useful with consecutive SH for each splat.
        auto shColor = EvalSphericalHarmonics(
            g_global_args.sphericalHarmonicsDegree, rayDir, &g_global_args.sphericalHarmonics[shIdx], blockDim.x);

        g_global_args.colors[index] = float4{shColor.x, shColor.y, shColor.z, opacity};
    }
}

void EvaluateSphericalHarmonics(CudaTimer& timer, int32_t count)
{
    // Each block processes 256 splats.
    // Dispatch as many groups as required.
    constexpr int32_t threadPerBlock{256};
    const int32_t numBlocks{(count + threadPerBlock - 1) / threadPerBlock};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(numBlocks)};

    timer.Start();
    EvaluateSphericalHarmonicsKernel<<<dimGrid, dimBlock>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateSphericalHarmonics\n");
    }
}

// Infer clip space data from world space data for each splat.
__global__ void EvaluateSplatClipDataKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < g_global_args.primitiveCount)
    {
        auto positionData = LoadReadOnly(&g_global_args.positions[index]);
        auto scaleData = LoadReadOnly(&g_global_args.scales[index]);
        auto rotationData = LoadReadOnly(&g_global_args.rotations[index]);

        auto position = glm::vec3(positionData.x, positionData.y, positionData.z);

        // Evaluate covariance.
        auto scale3x3 = glm::mat3(0);
        scale3x3[0][0] = scaleData.x;
        scale3x3[1][1] = scaleData.y;
        scale3x3[2][2] = scaleData.z;

        auto rotation = glm::quat::wxyz(rotationData.w, rotationData.x, rotationData.y, rotationData.z);

        auto RS = glm::mat3_cast(rotation) * scale3x3;
        auto splatCovariance = RS * glm::transpose(RS);

        // Centroid of the gaussian in view space.
        auto viewPosition = (glm::vec3)(g_global_args.cameraData.view * glm::vec4(position, 1));
        // We precompute camera related variables such as cotangents and Z parameters.
        auto fovCotangent = g_global_args.cameraData.fovCotangent;
        // We map the Z axis linearly, similarly to an orthographic projection.
        // This gives us better depth precision for sorting tiles later.
        auto depthScaleBias = g_global_args.cameraData.depthScaleBias;

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
        auto viewProjection = (glm::mat3) affineProjection * (glm::mat3) g_global_args.cameraData.view;
        // Evaluate covariance in clip space.
        auto clipCovariance = viewProjection * glm::transpose(splatCovariance) * glm::transpose(viewProjection);
        auto clipPosition = (glm::vec3)(affineProjection * glm::vec4(viewPosition, 1));

        // Tiny bump proportional to the area of a texel in clip space.
        // So that each splat at least cover a pixel.
        // Otherwise splats would disappear as they move away from the camera.
        // Too small for being hit by texel rays during rasterization.
        constexpr float kPi{3.14159265359};
        constexpr float kTexelSizeClip{2.0f / (float) kScreenSize};
        // Note: ellipse area is Pi * sqrt(det(cov)).
        constexpr float kTraceBump{(1.0f / kPi) * kTexelSizeClip * kTexelSizeClip};
        clipCovariance[0][0] += kTraceBump;
        clipCovariance[1][1] += kTraceBump;
        // Evaluate the eigen decomposition of the 2D covariance matrix to obtain
        // an oriented bounding rectangle for the splat.
        auto det = clipCovariance[0][0] * clipCovariance[1][1] - clipCovariance[1][0] * clipCovariance[1][0];
        // Trace over two.
        auto mid = 0.5f * (clipCovariance[0][0] + clipCovariance[1][1]);
        // Compute eigen values, see
        // https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html.
        constexpr float epsilon = 1e-12f;
        auto radius = glm::sqrt(glm::max(epsilon, mid * mid - det));
        auto lambda0 = mid + radius;
        // Max at zero to prevent computation errors (NaN), leads to a zero sized rect, tested to discard. the splat.
        auto lambda1 = glm::max(0.0f, mid - radius);

        auto eigenVector0 = glm::normalize(glm::vec2(clipCovariance[1][0], lambda0 - clipCovariance[0][0]));
        // Note that we use camera aspect to straighten the rectangle, aka obtain perpendicular vectors.
        auto eigenVector1 =
            glm::normalize(glm::vec2(eigenVector0.y / g_global_args.cameraData.aspect, -eigenVector0.x));

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

        // TODO: Ideally we'd do less work on out-of-frustum splats.
        // Out of frustum could be discarded as soon as we know its position.
        // No point in evaluating extent, conic, once we know the second eigenvalue is <= 0.
        // But we also want to retain coalesced memory accesses and avoid divergence.

        // Discard splats that are out of frustum.
        // Right now they'll be pulled when building the list, then they'll be found to cover zero tiles.
        auto edge = glm::step(glm::vec3(-1.0f), clipPosition) * glm::step(clipPosition, glm::vec3(1.0f));
        // Is visible: is it in frustum and the area is strictly positive?
        auto isVisible = edge.x * edge.y * edge.z * glm::step(0.0f, lambda1);
        // The constant is arbitrary.
        clipPosition = glm::mix(glm::vec3(-128.0f), clipPosition, isVisible);
        extent *= isVisible;

        // Global writes.
        g_global_args.positionsClipSpaceXY[index] = float2{clipPosition.x, clipPosition.y};
        g_global_args.positionsClipSpaceZ[index] = clipPosition.z;
        g_global_args.screenEllipses[index] = float4{glm::cos(angle), glm::sin(angle), extent.x, extent.y};
        g_global_args.conics[index] = float4{conic.x, conic.y, conic.z, 0.0f};
    }
}

void EvaluateSplatClipData(CudaTimer& timer, int32_t count)
{
    // Each block processes 256 splats.
    // Dispatch as many groups as required.
    constexpr int32_t threadPerBlock{256};
    const int32_t numBlocks{(count + threadPerBlock - 1) / threadPerBlock};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(numBlocks)};

    timer.Start();
    EvaluateSplatClipDataKernel<<<dimGrid, dimBlock>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateSplatsClipData\n");
    }
}

// Convert a point to ellipse local coordinates.
__device__ __forceinline__ glm::vec2 ConvertToEllipseCoordinates(const Ellipse& ellipse, const glm::vec2& point)
{
    // First handle translation.
    auto delta = point - ellipse.center;
    // Then rotation and scale.
    return glm::vec2((delta.x * ellipse.cosSin.x + delta.y * ellipse.cosSin.y) / ellipse.extent.x,
                     (delta.y * ellipse.cosSin.x - delta.x * ellipse.cosSin.y) / ellipse.extent.y);
}

// Test the intersection between a segment {v0, v1} and the unit circle.
__device__ __forceinline__ bool IntersectsUnitCircle(const glm::vec2& v0, const glm::vec2& v1)
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
__device__ __forceinline__ bool EllipseRectOverlap(const Ellipse& ellipse, const Rect& rect)
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
    overlaps |= glm::length2(ConvertToEllipseCoordinates(ellipse, rect.GetCenter())) < 1.0f;

    // Convert rectangle vertices to local ellipse coordinates.
    glm::vec2 points[4];
    points[0] = ConvertToEllipseCoordinates(ellipse, rect.min);
    points[1] = ConvertToEllipseCoordinates(ellipse, glm::vec2(rect.max.x, rect.min.y));
    points[2] = ConvertToEllipseCoordinates(ellipse, rect.max);
    points[3] = ConvertToEllipseCoordinates(ellipse, glm::vec2(rect.min.x, rect.max.y));

    // Test whether rectangle edges intersect with the ellipse.
    overlaps |= IntersectsUnitCircle(points[0], points[1]);
    overlaps |= IntersectsUnitCircle(points[1], points[2]);
    overlaps |= IntersectsUnitCircle(points[2], points[3]);
    overlaps |= IntersectsUnitCircle(points[3], points[0]);

    return overlaps;
}

// Evaluates the axis aligned bounding box of an ellipse.
__device__ __forceinline__ Rect GetAabbRect(const Ellipse& ellipse)
{
    auto right = ellipse.GetPrincipalAxis();
    auto up = ellipse.GetMinorAxis();

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

constexpr uint32_t kWarpMask{0xffffffff};
constexpr int32_t kWarpSize{32};
constexpr int32_t kWarpHalfSize{kWarpSize / 2};
constexpr int32_t kBuildTileListsThreadsPerGroup{kWarpSize * 8};
constexpr uint32_t kMaxUint32{std::numeric_limits<uint32_t>::max()};

// Clip depth in range [-1, 1]. (OpenGL convention.)
// Combine tile index and clip depth to form a sorting key.
__device__ __forceinline__ uint64_t GetSortKey(int32_t tile_index, float clipDepth)
{
    // Recall that we have hardcoded the final buffer size, 1024x1024.
    // 1024 / 16 = 64 -> 4096 tiles total, 12bits.
    // Our projection linearly maps [near, far] to [-1, 1].
    // We use the full 32 bits for maximum precision.
    auto quantizedDepth = (uint32_t) (glm::clamp((clipDepth + 1.0f) * 0.5f) * kMaxUint32);
    return ((uint64_t) tile_index << 32) | quantizedDepth;
}

__global__ void BuildTileListKernel()
{
    // Track splats chunks.
    __shared__ int32_t s_SplatStartIndex;
    __shared__ int32_t s_SplatCount;

    // Capacity matches one warp, the first warp which is responsible for loading splats.
    __shared__ int4 s_TilesRects[kWarpSize];
    __shared__ float4 s_ScreenEllipse[kWarpSize];
    __shared__ float2 s_PositionClipSpaceXY[kWarpSize];
    __shared__ float s_Depths[kWarpSize];
    __shared__ int s_TileExclusiveScan[kWarpSize];

    // Capacity matches the number of threads in the group.
    __shared__ int s_ExpandedTiles[kBuildTileListsThreadsPerGroup];
    __shared__ int s_ExpandedTileCount;
    __shared__ int s_TotalTilesCount;
    __shared__ int s_CumulatedExpandedTileCount;
    __shared__ bool s_HasPendingTiles;

    __shared__ int64_t s_TileKeys[kBuildTileListsThreadsPerGroup];
    __shared__ int32_t s_TileValues[kBuildTileListsThreadsPerGroup];
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
            auto splatCounter = atomicAdd(&g_splat_counter, kWarpSize);
            s_SplatStartIndex = glm::min(splatCounter, g_global_args.primitiveCount);
            auto splatEndIndex = glm::min(splatCounter + kWarpSize, g_global_args.primitiveCount);
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

        // The first warp pulls a chunk of splats and updates the tile list.
        if (threadIdx.x < kWarpSize)
        {
            // Read splats data, coalesced.
            // Only threads from the first warp may enter here.
            if (threadIdx.x < s_SplatCount)
            {
                // Read current splat data. The one this thread is responsible for.
                auto srcIndex = s_SplatStartIndex + threadIdx.x;
                auto positionClipSpaceXYData = LoadReadOnly(&g_global_args.positionsClipSpaceXY[srcIndex]);
                auto ellipseData = LoadReadOnly(&g_global_args.screenEllipses[srcIndex]);

                auto ellipse = Ellipse();
                ellipse.center = glm::vec2(positionClipSpaceXYData.x, positionClipSpaceXYData.y);
                ellipse.cosSin = glm::vec2(ellipseData.x, ellipseData.y);
                ellipse.extent = glm::vec2(ellipseData.z, ellipseData.w);

                auto rect = GetAabbRect(ellipse);

                // From clip to tiles.
                auto tilesRectFloat = (glm::vec4(rect.min, rect.max) + 1.0f) * 0.5f * (float) kTilesPerScreen;

                // Clip within screen bounds.
                auto tilesRect = glm::ivec4(
                    glm::clamp((int32_t) glm::floor(tilesRectFloat.x), 0, kTilesPerScreen),
                    glm::clamp((int32_t) glm::floor(tilesRectFloat.y), 0, kTilesPerScreen),
                    glm::clamp((int32_t) glm::ceil(tilesRectFloat.z), 0, kTilesPerScreen),
                    glm::clamp((int32_t) glm::ceil(tilesRectFloat.w), 0, kTilesPerScreen));

                // Min-Max to Center-Size representation.
                tilesRect.zw -= tilesRect.xy;

                // We must account for negative surface of the rectangle.
                // Note: We could prevent this earlier. But it would require a discard mechnism earlier in the pipeline.
                tileCount = glm::max(0, tilesRect.z * tilesRect.w);

                // Store shared data.
                s_Depths[threadIdx.x] = LoadReadOnly(&g_global_args.positionsClipSpaceZ[srcIndex]);
                s_TilesRects[threadIdx.x] = int4{tilesRect.x, tilesRect.y, tilesRect.z, tilesRect.w};
                s_ScreenEllipse[threadIdx.x] = ellipseData;
                s_PositionClipSpaceXY[threadIdx.x] = positionClipSpaceXYData;
            }

            // What if no thread has tiles to process? (All splats out of frustum.)
            auto hasTiles = __ballot_sync(kWarpMask, tileCount != 0) != 0;

            if (hasTiles)
            {
                // Only the first warp passes here, so threadIdx.x is the lane.
                auto inclusiveScan = tileCount;
#pragma unroll
                for (auto delta = 1u; delta <= kWarpHalfSize; delta <<= 1u)
                {
                    auto n = __shfl_up_sync(kWarpMask, inclusiveScan, delta);

                    // Only process the threads that do have some work.
                    // We never clear the shared array, don't touch what lingers from previous runs.
                    // Note: first warp -> threadIdx == laneId.
                    if (threadIdx.x >= delta)
                    {
                        inclusiveScan += n;
                    }
                }

                s_TileExclusiveScan[threadIdx.x] = inclusiveScan - tileCount;

                if (threadIdx.x == kWarpSize - 1)
                {
                    s_TotalTilesCount = inclusiveScan;
                }

                // Each thread writes its tiles in shared memory, to be processed later by the whole group.
                // Exclusive scan allows each thread to evaluate its writing indices in shared memory.
                auto startWriteIndex = glm::min(inclusiveScan - tileCount, kBuildTileListsThreadsPerGroup);
                auto endWriteIndex = glm::min(inclusiveScan, kBuildTileListsThreadsPerGroup);
                auto writeIndex = startWriteIndex;

                // Write tiles in shared memory.
                while (__ballot_sync(kWarpMask, writeIndex < endWriteIndex) != 0)
                {
                    if (writeIndex < endWriteIndex)
                    {
                        s_ExpandedTiles[writeIndex] = threadIdx.x;
                        ++writeIndex;
                    }
                }

                __syncwarp();

                // TODO: What if there are no tiles for any thread?
                // TODO: Conditional breakpoint -> NO tile during pass. -> s_TotalTilesCount == 0
                // The last thread writing to shared memory is responsible for the tracking of shared counters.
                if (endWriteIndex - startWriteIndex > 0)
                {
                    // TODO: If zero tile, cause we never enter here, bookkeeping is out of sync.
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
                    else if (endWriteIndex == kBuildTileListsThreadsPerGroup)
                    {
                        s_HasPendingTiles = true;
                        s_ExpandedTileCount = endWriteIndex;
                        s_CumulatedExpandedTileCount = s_ExpandedTileCount;
                    }
                }
            }
        }

        // Wait for shared memory updates to be visible to the whole group.
        __syncthreads();

        if (s_ExpandedTileCount == 0)
        {
            // Nothing to do, move on and pull a new chunk of splats.
            continue;
        }

        // Test and coommit tiles, the whole group works.
        for (;;)
        {
            // Overlap test between a tile and an ellipse corresponding to the outline of a splat.
            if (threadIdx.x < s_ExpandedTileCount)
            {
                auto splatIndex = s_ExpandedTiles[threadIdx.x];
                auto tilesRect = BuiltinToGlmVec4i(s_TilesRects[splatIndex]);
                auto localTileIndex =
                    threadIdx.x - s_TileExclusiveScan[splatIndex] + s_CumulatedExpandedTileCount - s_ExpandedTileCount;

                // Coordinates of the current tile within the splat AABB rect.
                auto localTileCoords = glm::ivec2(localTileIndex % tilesRect.z, localTileIndex / tilesRect.z);
                // Coordinates of the current tile within the screen.
                auto globalTileCoords = tilesRect.xy + localTileCoords;
                constexpr float tileNormalizedSize = kTileSize / (float) kScreenSize;
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

                if (EllipseRectOverlap(ellipse, tileRectClipSpace))
                {
                    // The insertion index must be valid since there's room for every thread in the group.
                    auto localInsertionIndex = atomicAdd(&s_TileIndex, 1);
                    auto globalTileIndex = globalTileCoords.y * kTilesPerScreen + globalTileCoords.x;
                    s_TileKeys[localInsertionIndex] = GetSortKey(globalTileIndex, s_Depths[splatIndex]);
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
                    auto tileTileListCounter = atomicAdd(&g_tile_counter, s_TileIndex);
                    // Restrict to target capacity.
                    s_WriteTileStartIndex = glm::min(tileTileListCounter, g_tile_list_args.capacity);
                    s_WriteTileEndIndex = glm::min(tileTileListCounter + s_TileIndex, g_tile_list_args.capacity);
                    // Reset.
                    s_TileIndex = 0;
                }

                __syncthreads();

                // Abort and exit if global memory is full.
                if (s_WriteTileEndIndex == g_tile_list_args.capacity)
                {
                    return;
                }

                // Write global memory.
                if (threadIdx.x < s_WriteTileEndIndex - s_WriteTileStartIndex)
                {
                    auto dstIndex = s_WriteTileStartIndex + threadIdx.x;
                    g_tile_list_args.keys[dstIndex] = s_TileKeys[threadIdx.x];
                    g_tile_list_args.values[dstIndex] = s_TileValues[threadIdx.x];
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
            if (threadIdx.x < kWarpSize)
            {
                // Evaluate the tiles pending processing for the current splat.
                auto exclusiveScan = s_TileExclusiveScan[threadIdx.x] - s_CumulatedExpandedTileCount;
                auto startWriteIndex = glm::clamp(exclusiveScan, 0, kBuildTileListsThreadsPerGroup);
                auto endWriteIndex = glm::clamp(exclusiveScan + tileCount, 0, kBuildTileListsThreadsPerGroup);
                auto writeIndex = startWriteIndex;

                // Write the next batch of tiles to shared memory.
                while (__ballot_sync(kWarpMask, writeIndex < endWriteIndex) != 0)
                {
                    if (writeIndex < endWriteIndex)
                    {
                        s_ExpandedTiles[writeIndex] = threadIdx.x;
                        ++writeIndex;
                    }
                }

                __syncwarp();

                // TODO: If no tile who's the last thread?
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
                    else if (endWriteIndex == kBuildTileListsThreadsPerGroup)
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
static int32_t initCounter{0};

// Emit tiles per splat.
int32_t BuildTileList(CudaTimer& timer, int32_t numBlocks, int32_t tileListCapacity)
{
    // Reset counters. (Splats and tile render list.)
    initCounter = 0;
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(g_splat_counter, &initCounter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(g_tile_counter, &initCounter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));

    // Note that number of groups is passed by the caller.
    // Recall that we have persistent group: they run as long as there is remaining work.
    // This is why the workload is not considered for dispatch parameters.
    const auto dimBlock = dim3(kBuildTileListsThreadsPerGroup);
    const auto dimGrid = dim3(numBlocks);

    timer.Start();
    BuildTileListKernel<<<dimGrid, dimBlock>>>();
    timer.Stop();

    CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(&initCounter, g_tile_counter, sizeof(int32_t), 0, cudaMemcpyDeviceToHost));

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error buildTileList\n");
    }

    // Minimum since atomicAdd(...) has no rollback in case of excess.
    return glm::min(initCounter, tileListCapacity);
}

// Sort the list of tiles.
// Done with cub::, which requires a temporary buffer that will be reallocated as needed.
void SortTileList(CudaTimer& timer,
                  int32_t tileListSize,
                  void*& deviceTempStorage,
                  size_t& tempStorageSizeInBytes,
                  DoubleBuffer<uint64_t>& keys,
                  DoubleBuffer<int32_t>& values)
{
    timer.Start();

    // We can skip the top bits in the key.
    // 32 bits for depth, 12 bits for tile index.
    // (Depends on the number of tiles on screen, which is hardcoded.)
    constexpr int32_t beginBit{0};
    constexpr int32_t endBit{32 + 12};

    // Instantiate cub:: double buffers.
    cub::DoubleBuffer<uint64_t> cubKeys(keys.Current(), keys.Alternate());
    cub::DoubleBuffer<int32_t> cubValues(values.Current(), values.Alternate());

    // Dry run. Allocate temp memory if needed. From the documentation about "d_temp_storage":
    // "When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done."
    size_t requiredTempStorageSizeInBytes;
    cub::DeviceRadixSort::SortPairs(
        (void*) nullptr, requiredTempStorageSizeInBytes, cubKeys, cubValues, tileListSize, beginBit, endBit);

    if (requiredTempStorageSizeInBytes > tempStorageSizeInBytes)
    {
        if (deviceTempStorage != nullptr)
        {
            CHECK_CUDA_ERRORS(cudaFree(deviceTempStorage));
        }
        CHECK_CUDA_ERRORS(cudaMalloc((void**) &deviceTempStorage, requiredTempStorageSizeInBytes));
        tempStorageSizeInBytes = requiredTempStorageSizeInBytes;
    }

    // Actual sorting.
    cub::DeviceRadixSort::SortPairs(
        deviceTempStorage, tempStorageSizeInBytes, cubKeys, cubValues, tileListSize, beginBit, endBit);

    // Copy double buffers back to host, so that we know which buffer to use from now on.
    keys = DoubleBuffer<uint64_t>(cubKeys.Current(), cubKeys.Alternate());
    values = DoubleBuffer<int32_t>(cubValues.Current(), cubValues.Alternate());

    timer.Stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("error sortTileList\n");
    }
}

__global__ void EvaluateTileRangesKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    // The first thread within the grid handles first and last elements.
    // Not that bad, only one warp within the grid will have divergence.
    if (index == 0)
    {
        auto firstKey = LoadReadOnly(&g_tile_list_args.keys[0]);
        auto firstTileIndex = firstKey >> 32;
        g_global_args.tileRanges[firstTileIndex * 2] = 0;

        auto lastKey = LoadReadOnly(&g_tile_list_args.keys[g_tile_list_args.size - 1]);
        auto lastTileIndex = lastKey >> 32;
        g_global_args.tileRanges[lastTileIndex * 2 + 1] = g_tile_list_args.size;
    }
    else if (index < g_tile_list_args.size)
    {
        // Pairwise comparison, look at the previous element.
        auto prevKey = LoadReadOnly(&g_tile_list_args.keys[index - 1]);
        auto prevTileIndex = prevKey >> 32;
        auto key = LoadReadOnly(&g_tile_list_args.keys[index]);
        auto tileIndex = key >> 32;

        // If we detect a tile change,
        // update the end of the previous range and the start of the current one.
        if (tileIndex != prevTileIndex)
        {
            g_global_args.tileRanges[prevTileIndex * 2 + 1] = index;
            g_global_args.tileRanges[tileIndex * 2] = index;
        }
    }
}

// Evaluate ranges within the list corresponding to each tile.
void EvaluateTileRange(CudaTimer& timer, int32_t tileListSize)
{
    constexpr int32_t threadPerBlock{256};
    const int32_t numBlocks{(tileListSize + threadPerBlock - 1) / threadPerBlock};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(numBlocks)};

    timer.Start();
    EvaluateTileRangesKernel<<<dimGrid, dimBlock>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateTileRanges\n");
    }
}

__global__ void RasterizeTilesKernel()
{
    __shared__ int32_t s_FirstSplatIndex;
    __shared__ int32_t s_SplatsCount;
    __shared__ float4 s_Colors[kWarpSize];
    __shared__ float4 s_Conics[kWarpSize];
    __shared__ float2 s_Centers[kWarpSize];

    // The first thread pulls the range beginning and end.
    if (threadIdx.x == 0)
    {
        // Block Id corresponds to the tile.
        s_FirstSplatIndex = LoadReadOnly(&g_global_args.tileRanges[blockIdx.x * 2]);
        auto endSplatIndex = LoadReadOnly(&g_global_args.tileRanges[blockIdx.x * 2 + 1]);
        s_SplatsCount = endSplatIndex - s_FirstSplatIndex;
    }

    __syncthreads();

    // If the range is empty, exit.
    if (s_SplatsCount == 0)
    {
        return;
    }

    // Coordinates of the pixel handled by this thread within the tile .
    const auto threadTilePixelCoords = glm::ivec2(threadIdx.x % kTileSize, threadIdx.x / kTileSize);
    // Coordinates of the pixel within the assembled image.
    const auto threadBufferPixelCoords =
        glm::ivec2(blockIdx.x % kTilesPerScreen, blockIdx.x / kTilesPerScreen) * kTileSize + threadTilePixelCoords;
    // TODO: Account for pixel center offset (0.5, 0.5)?
    const auto clipCoords = (glm::vec2) threadBufferPixelCoords * (2.0f / (float) kScreenSize) - 1.0f;

    // Initialize color and transmittance for blending. (Volumetric Rendering.)
    auto color = glm::vec3(0.0f);
    auto transmittance = 1.0f;

    while (s_SplatsCount > 0)
    {
        // Don't overflow shared buffers.
        // We only load one 32 splats chunk at once, in hopes of needing no more.
        // These loads aren't coalesced.
        auto splatsCount = glm::min(s_SplatsCount, kWarpSize);

        __syncthreads();

        // Load splat data.
        if (threadIdx.x < splatsCount)
        {
            auto srcIndex = g_tile_list_args.values[s_FirstSplatIndex + threadIdx.x];
            s_Centers[threadIdx.x] = LoadReadOnly(&g_global_args.positionsClipSpaceXY[srcIndex]);
            auto conicData = LoadReadOnly(&g_global_args.conics[srcIndex]);
            auto colorData = LoadReadOnly(&g_global_args.colors[srcIndex]);
            s_Colors[threadIdx.x] = float4{colorData.x, colorData.y, colorData.z, colorData.w};
            s_Conics[threadIdx.x] = float4{conicData.x, conicData.y, conicData.z, 0.0f};
        }

        // Wait for loaded data to be visible to all threads.
        __syncthreads();

        // Rasterize the current chunk.
        for (auto i = 0; i != splatsCount; ++i)
        {
            // Square distance to splat center.
            auto d = clipCoords - BuiltinToGlmVec2(s_Centers[i]);
            auto splatColor = BuiltinToGlmVec4(s_Colors[i]);
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

    /*
    color.x = glm::pow(color.x, 2.2f);
    color.y = glm::pow(color.y, 2.2f);
    color.z = glm::pow(color.z, 2.2f);
    */

    uchar4 quantizedColor;
    quantizedColor.x = color.x * 255;
    quantizedColor.y = color.y * 255;
    quantizedColor.z = color.z * 255;
    quantizedColor.w = 255;

    // Commit to global memory.
    auto globalWriteIndex = threadBufferPixelCoords.y * kScreenSize + threadBufferPixelCoords.x;
    g_global_args.backBuffer[globalWriteIndex] = quantizedColor;
}

// Rasterize tiles.
void RasterizeTile(CudaTimer& timer)
{
    // We dispatch one block per tile, one thread per pixel within the tile.
    constexpr int32_t threadPerBlock{kTileSize * kTileSize};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(kTotalTiles)};

    timer.Start();
    RasterizeTilesKernel<<<dimGrid, dimBlock>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error renderDepthBuffer\n");
    }
}
