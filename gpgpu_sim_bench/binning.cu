#include "common.cuh"
#include <stdio.h>

// Binning (splat -> tile list write-in):
// For each splat, compute the tiles overlapped by its ellipse, and append its index + key to per-tile arrays.

struct BinningOut
{
    int* tileCounts;          // [GSIM_TOTAL_TILES]
    int* tileValues;          // [GSIM_TOTAL_TILES * GSIM_MAX_SPLATS_PER_TILE]
    uint64_t* tileKeys;       // same shape as tileValues
};

__global__ void gsim_clear_tile_counts(int* tileCounts)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < (int)GSIM_TOTAL_TILES) tileCounts[i] = 0;
}

__global__ void gsim_binning_kernel(
    int splatCount,
    const float2* positionClipXY,
    const float* positionClipZ,
    const float4* screenEllipse, // (cos,sin,ex,ey)
    BinningOut out)
{
    int splatId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (splatId >= splatCount) return;

    float2 c = positionClipXY[splatId];
    float z = positionClipZ[splatId];
    float4 e = screenEllipse[splatId];

    // Invisible splats have extent 0 or center moved out.
    if (e.z <= 0.0f || e.w <= 0.0f) return;

    GsimEllipse ellipse;
    ellipse.center = c;
    ellipse.cosSin = make_float2(e.x, e.y);
    ellipse.extent = make_float2(e.z, e.w);

    // Conservative AABB in clip-space using max extent.
    float r = (ellipse.extent.x > ellipse.extent.y) ? ellipse.extent.x : ellipse.extent.y;

    float minx = clampf(c.x - r, -1.0f, 1.0f);
    float maxx = clampf(c.x + r, -1.0f, 1.0f);
    float miny = clampf(c.y - r, -1.0f, 1.0f);
    float maxy = clampf(c.y + r, -1.0f, 1.0f);

    // Convert clip to tile coords.
    float fx0 = (minx + 1.0f) * 0.5f * (float)GSIM_TILES_PER_SCREEN;
    float fx1 = (maxx + 1.0f) * 0.5f * (float)GSIM_TILES_PER_SCREEN;
    float fy0 = (miny + 1.0f) * 0.5f * (float)GSIM_TILES_PER_SCREEN;
    float fy1 = (maxy + 1.0f) * 0.5f * (float)GSIM_TILES_PER_SCREEN;

    int tx0 = (int)floorf(fx0);
    int tx1 = (int)ceilf(fx1);
    int ty0 = (int)floorf(fy0);
    int ty1 = (int)ceilf(fy1);

    if (tx0 < 0) tx0 = 0;
    if (ty0 < 0) ty0 = 0;
    if (tx1 > (int)GSIM_TILES_PER_SCREEN) tx1 = (int)GSIM_TILES_PER_SCREEN;
    if (ty1 > (int)GSIM_TILES_PER_SCREEN) ty1 = (int)GSIM_TILES_PER_SCREEN;

    for (int ty = ty0; ty < ty1; ++ty)
    {
        for (int tx = tx0; tx < tx1; ++tx)
        {
            int tileId = ty * (int)GSIM_TILES_PER_SCREEN + tx;
            GsimRect rect = tile_rect_clip(tx, ty);
            if (!ellipse_rect_overlap(ellipse, rect))
            {
                continue;
            }

            int idx = atomicAdd(&out.tileCounts[tileId], 1);
            if (idx < (int)GSIM_MAX_SPLATS_PER_TILE)
            {
                out.tileValues[tileId * (int)GSIM_MAX_SPLATS_PER_TILE + idx] = splatId;
                out.tileKeys[tileId * (int)GSIM_MAX_SPLATS_PER_TILE + idx] = gsim_make_key(tileId, z);
            }
        }
    }
}

extern "C" void gsim_binning(
    int splatCount,
    const float2* d_positionClipXY,
    const float* d_positionClipZ,
    const float4* d_screenEllipse,
    int* d_tileCounts,
    int* d_tileValues,
    uint64_t* d_tileKeys)
{
    const int threads = 256;
    const int blocksTiles = ((int)GSIM_TOTAL_TILES + threads - 1) / threads;
    gsim_clear_tile_counts<<<blocksTiles, threads>>>(d_tileCounts);

    BinningOut out;
    out.tileCounts = d_tileCounts;
    out.tileValues = d_tileValues;
    out.tileKeys = d_tileKeys;

    const int blocks = (splatCount + threads - 1) / threads;
    gsim_binning_kernel<<<blocks, threads>>>(splatCount, d_positionClipXY, d_positionClipZ, d_screenEllipse, out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("gsim_binning kernel error: %s\n", cudaGetErrorString(err));
    }
}
