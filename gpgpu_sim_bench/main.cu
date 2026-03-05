#include "common.cuh"

#include <stdio.h>
#include <stdlib.h>

// Declarations from other compilation units.
struct Cov2;
extern "C" void gsim_preprocess(
    int splatCount,
    const float2* d_inClipXY,
    const float* d_inClipZ,
    const Cov2* d_inCov,
    float2* d_outClipXY,
    float* d_outClipZ,
    float4* d_outScreenEllipse,
    float4* d_outConic,
    float sigmaScale);

extern "C" void gsim_binning(
    int splatCount,
    const float2* d_positionClipXY,
    const float* d_positionClipZ,
    const float4* d_screenEllipse,
    int* d_tileCounts,
    int* d_tileValues,
    uint64_t* d_tileKeys);

extern "C" void gsim_rasterize(
    const int* d_tileCounts,
    const int* d_tileValues,
    const uint64_t* d_tileKeys,
    const float2* d_positionClipXY,
    const float4* d_conic,
    const float4* d_color,
    uchar4* d_backBuffer);

// Build tip (example):
//   nvcc -O3 -lineinfo -I. main.cu screen_preprocess.cu binning.cu tile_raster.cu -o gsim_bench
// For GPGPU-Sim you typically compile to PTX and run with its config.

static float frand01(unsigned int* state)
{
    // xorshift32
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (float)(x & 0x00FFFFFFu) / (float)0x01000000u;
}

int main(int argc, char** argv)
{
    int splatCount = 32768;
    if (argc > 1) splatCount = atoi(argv[1]);
    if (splatCount < 1) splatCount = 1;
    if (splatCount > (int)GSIM_MAX_SPLATS) splatCount = (int)GSIM_MAX_SPLATS;

    printf("GSIM bench: splats=%d, screen=%dx%d, tiles=%dx%d, maxPerTile=%d\n",
        splatCount,
        (int)GSIM_SCREEN_SIZE, (int)GSIM_SCREEN_SIZE,
        (int)GSIM_TILES_PER_SCREEN, (int)GSIM_TILES_PER_SCREEN,
        (int)GSIM_MAX_SPLATS_PER_TILE);

    // Host-side synthetic inputs.
    float2* h_clipXY = (float2*)malloc(sizeof(float2) * (size_t)splatCount);
    float* h_clipZ = (float*)malloc(sizeof(float) * (size_t)splatCount);
    Cov2* h_cov = (Cov2*)malloc(sizeof(Cov2) * (size_t)splatCount);
    float4* h_color = (float4*)malloc(sizeof(float4) * (size_t)splatCount);

    unsigned int rng = 0xC0FFEEu;
    for (int i = 0; i < splatCount; ++i)
    {
        // Random center mostly in frustum.
        float x = frand01(&rng) * 2.0f - 1.0f;
        float y = frand01(&rng) * 2.0f - 1.0f;
        float z = frand01(&rng) * 2.0f - 1.0f;
        h_clipXY[i] = make_float2(x, y);
        h_clipZ[i] = z;

        // Random positive-definite covariance in clip space.
        // Keep small so binning doesn't explode.
        float sx = 0.002f + frand01(&rng) * 0.02f;
        float sy = 0.002f + frand01(&rng) * 0.02f;
        float rho = (frand01(&rng) * 2.0f - 1.0f) * 0.2f;
        Cov2 c;
        c.a = sx * sx;
        c.c = sy * sy;
        c.b = rho * sx * sy;
        h_cov[i] = c;

        float r = frand01(&rng);
        float g = frand01(&rng);
        float b = frand01(&rng);
        float a = 0.05f + frand01(&rng) * 0.45f;
        h_color[i] = make_float4(r, g, b, a);
    }

    // Device allocations.
    float2 *d_inClipXY = 0, *d_clipXY = 0;
    float *d_inClipZ = 0, *d_clipZ = 0;
    Cov2* d_cov = 0;
    float4* d_screenEllipse = 0;
    float4* d_conic = 0;
    float4* d_color = 0;

    int* d_tileCounts = 0;
    int* d_tileValues = 0;
    uint64_t* d_tileKeys = 0;

    uchar4* d_backBuffer = 0;

    cudaMalloc((void**)&d_inClipXY, sizeof(float2) * (size_t)splatCount);
    cudaMalloc((void**)&d_inClipZ, sizeof(float) * (size_t)splatCount);
    cudaMalloc((void**)&d_cov, sizeof(Cov2) * (size_t)splatCount);

    cudaMalloc((void**)&d_clipXY, sizeof(float2) * (size_t)splatCount);
    cudaMalloc((void**)&d_clipZ, sizeof(float) * (size_t)splatCount);
    cudaMalloc((void**)&d_screenEllipse, sizeof(float4) * (size_t)splatCount);
    cudaMalloc((void**)&d_conic, sizeof(float4) * (size_t)splatCount);

    cudaMalloc((void**)&d_color, sizeof(float4) * (size_t)splatCount);

    cudaMalloc((void**)&d_tileCounts, sizeof(int) * (size_t)GSIM_TOTAL_TILES);
    cudaMalloc((void**)&d_tileValues, sizeof(int) * (size_t)GSIM_TOTAL_TILES * (size_t)GSIM_MAX_SPLATS_PER_TILE);
    cudaMalloc((void**)&d_tileKeys, sizeof(uint64_t) * (size_t)GSIM_TOTAL_TILES * (size_t)GSIM_MAX_SPLATS_PER_TILE);

    cudaMalloc((void**)&d_backBuffer, sizeof(uchar4) * (size_t)GSIM_SCREEN_SIZE * (size_t)GSIM_SCREEN_SIZE);

    cudaMemcpy(d_inClipXY, h_clipXY, sizeof(float2) * (size_t)splatCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inClipZ, h_clipZ, sizeof(float) * (size_t)splatCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cov, h_cov, sizeof(Cov2) * (size_t)splatCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_color, h_color, sizeof(float4) * (size_t)splatCount, cudaMemcpyHostToDevice);

    cudaMemset(d_backBuffer, 0, sizeof(uchar4) * (size_t)GSIM_SCREEN_SIZE * (size_t)GSIM_SCREEN_SIZE);

    // Run stages.
    GsimTimers timer;
    gsim_timer_init(&timer);

    gsim_timer_begin(&timer);
    gsim_preprocess(splatCount, d_inClipXY, d_inClipZ, d_cov, d_clipXY, d_clipZ, d_screenEllipse, d_conic, /*sigma*/3.0f);
    cudaDeviceSynchronize();
    float msPre = gsim_timer_end_ms(&timer);

    gsim_timer_begin(&timer);
    gsim_binning(splatCount, d_clipXY, d_clipZ, d_screenEllipse, d_tileCounts, d_tileValues, d_tileKeys);
    cudaDeviceSynchronize();
    float msBin = gsim_timer_end_ms(&timer);

    gsim_timer_begin(&timer);
    gsim_rasterize(d_tileCounts, d_tileValues, d_tileKeys, d_clipXY, d_conic, d_color, d_backBuffer);
    cudaDeviceSynchronize();
    float msRas = gsim_timer_end_ms(&timer);

    gsim_timer_destroy(&timer);

    // Simple checksum to prevent DCE and to sanity check determinism.
    unsigned int* h_hash = (unsigned int*)malloc(sizeof(unsigned int));
    unsigned int hash = 2166136261u;

    // Sample a subset of pixels (stride) for speed.
    const int stride = 257;
    uchar4* h_pixels = (uchar4*)malloc(sizeof(uchar4) * (size_t)GSIM_SCREEN_SIZE * (size_t)GSIM_SCREEN_SIZE);
    cudaMemcpy(h_pixels, d_backBuffer, sizeof(uchar4) * (size_t)GSIM_SCREEN_SIZE * (size_t)GSIM_SCREEN_SIZE, cudaMemcpyDeviceToHost);

    const int totalPx = (int)GSIM_SCREEN_SIZE * (int)GSIM_SCREEN_SIZE;
    for (int i = 0; i < totalPx; i += stride)
    {
        uchar4 p = h_pixels[i];
        hash ^= (unsigned int)p.x; hash *= 16777619u;
        hash ^= (unsigned int)p.y; hash *= 16777619u;
        hash ^= (unsigned int)p.z; hash *= 16777619u;
        hash ^= (unsigned int)p.w; hash *= 16777619u;
    }
    *h_hash = hash;

    printf("timing_ms preprocess=%.3f binning=%.3f raster=%.3f  checksum=0x%08X\n", msPre, msBin, msRas, hash);

    // Cleanup.
    free(h_clipXY);
    free(h_clipZ);
    free(h_cov);
    free(h_color);
    free(h_pixels);
    free(h_hash);

    cudaFree(d_inClipXY);
    cudaFree(d_inClipZ);
    cudaFree(d_cov);
    cudaFree(d_clipXY);
    cudaFree(d_clipZ);
    cudaFree(d_screenEllipse);
    cudaFree(d_conic);
    cudaFree(d_color);
    cudaFree(d_tileCounts);
    cudaFree(d_tileValues);
    cudaFree(d_tileKeys);
    cudaFree(d_backBuffer);

    return 0;
}
