#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Standalone CUDA micro-benchmark utilities (no GLM/OpenGL/CUB).

#ifndef GSIM_SCREEN_SIZE
#define GSIM_SCREEN_SIZE 1024
#endif

#ifndef GSIM_TILE_SIZE
#define GSIM_TILE_SIZE 16
#endif

#ifndef GSIM_TILES_PER_SCREEN
#define GSIM_TILES_PER_SCREEN (GSIM_SCREEN_SIZE / GSIM_TILE_SIZE)
#endif

#ifndef GSIM_TOTAL_TILES
#define GSIM_TOTAL_TILES (GSIM_TILES_PER_SCREEN * GSIM_TILES_PER_SCREEN)
#endif

// Maximum number of splats stored per tile for binning.
// Keep small for simulation runs; raise as needed.
#ifndef GSIM_MAX_SPLATS_PER_TILE
#define GSIM_MAX_SPLATS_PER_TILE 1024
#endif

// Maximum splats processed in a run (host allocation size).
#ifndef GSIM_MAX_SPLATS
#define GSIM_MAX_SPLATS 65536
#endif

// Basic float2/float4 helpers (avoid pulling extra deps).
__host__ __device__ __forceinline__ float2 make_f2(float x, float y) { return make_float2(x, y); }
__host__ __device__ __forceinline__ float4 make_f4(float x, float y, float z, float w) { return make_float4(x, y, z, w); }

__host__ __device__ __forceinline__ float2 f2_add(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__host__ __device__ __forceinline__ float2 f2_sub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__host__ __device__ __forceinline__ float2 f2_mul(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
__host__ __device__ __forceinline__ float f2_dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
__host__ __device__ __forceinline__ float f2_len2(float2 a) { return f2_dot(a, a); }

__host__ __device__ __forceinline__ float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

// Screen-space ellipse parameterization.
// center: clip-space XY in [-1,1]
// cosSin: (cos(theta), sin(theta))
// extent: ellipse radii in clip space
struct GsimEllipse
{
    float2 center;
    float2 cosSin;
    float2 extent;
};

struct GsimRect
{
    float2 min;
    float2 max;
};

// 2D covariance in screen/clip coordinates: [[a,b],[b,c]]
struct Cov2
{
    float a;
    float b;
    float c;
};

// Return (tileId<<32) | quantizedDepth, like the original renderer.
__host__ __device__ __forceinline__ uint64_t gsim_make_key(int tileId, float clipDepth)
{
    const uint32_t maxu = 0xFFFFFFFFu;
    float t = clampf((clipDepth + 1.0f) * 0.5f, 0.0f, 1.0f);
    uint32_t q = (uint32_t)(t * (float)maxu);
    return ((uint64_t)(uint32_t)tileId << 32) | (uint64_t)q;
}

__device__ __forceinline__ float2 ellipse_local_coords(const GsimEllipse& e, const float2& p)
{
    // translate
    float2 d = f2_sub(p, e.center);
    // rotate by [cos sin; -sin cos] and scale by extent
    float x = (d.x * e.cosSin.x + d.y * e.cosSin.y) / e.extent.x;
    float y = (d.y * e.cosSin.x - d.x * e.cosSin.y) / e.extent.y;
    return make_float2(x, y);
}

__device__ __forceinline__ bool segment_intersects_unit_circle(float2 v0, float2 v1)
{
    float2 d = f2_sub(v1, v0);
    float len2 = f2_len2(d);
    if (len2 <= 0.0f) return f2_len2(v0) < 1.0f;
    float t = __saturatef((-f2_dot(v0, d)) / len2);
    float2 proj = f2_add(v0, f2_mul(d, t));
    return f2_len2(proj) < 1.0f;
}

__device__ __forceinline__ bool ellipse_rect_overlap(const GsimEllipse& e, const GsimRect& r)
{
    // ellipse center in rect
    bool overlaps = (e.center.x > r.min.x) && (e.center.x < r.max.x) && (e.center.y > r.min.y) && (e.center.y < r.max.y);

    // rect center in ellipse
    float2 rc = make_float2(0.5f * (r.min.x + r.max.x), 0.5f * (r.min.y + r.max.y));
    overlaps = overlaps || (f2_len2(ellipse_local_coords(e, rc)) < 1.0f);

    // rect edges intersect unit circle in ellipse-local space
    float2 p0 = ellipse_local_coords(e, r.min);
    float2 p1 = ellipse_local_coords(e, make_float2(r.max.x, r.min.y));
    float2 p2 = ellipse_local_coords(e, r.max);
    float2 p3 = ellipse_local_coords(e, make_float2(r.min.x, r.max.y));

    overlaps = overlaps || segment_intersects_unit_circle(p0, p1);
    overlaps = overlaps || segment_intersects_unit_circle(p1, p2);
    overlaps = overlaps || segment_intersects_unit_circle(p2, p3);
    overlaps = overlaps || segment_intersects_unit_circle(p3, p0);

    return overlaps;
}

// Convert tile coords (tx,ty) to clip-space rectangle.
__host__ __device__ __forceinline__ GsimRect tile_rect_clip(int tx, int ty)
{
    const float tileNorm = (float)GSIM_TILE_SIZE / (float)GSIM_SCREEN_SIZE;
    const float tileClip = tileNorm * 2.0f;
    float2 minp = make_float2((float)tx * tileClip - 1.0f, (float)ty * tileClip - 1.0f);
    float2 maxp = make_float2(minp.x + tileClip, minp.y + tileClip);
    GsimRect r; r.min = minp; r.max = maxp; return r;
}

struct GsimTimers
{
    cudaEvent_t start;
    cudaEvent_t stop;
};

inline void gsim_timer_init(GsimTimers* t)
{
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

inline void gsim_timer_destroy(GsimTimers* t)
{
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

inline void gsim_timer_begin(GsimTimers* t)
{
    cudaEventRecord(t->start);
}

inline float gsim_timer_end_ms(GsimTimers* t)
{
    cudaEventRecord(t->stop);
    cudaEventSynchronize(t->stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}
