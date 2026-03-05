#include "common.cuh"
#include <stdio.h>

// Tile rasterization + alpha blending.
// One block per tile, one thread per pixel (GSIM_TILE_SIZE*GSIM_TILE_SIZE).
// Uses per-tile binned lists (tileCounts + tileValues + tileKeys), and per-splat attributes (center, conic, color).

struct RasterIn
{
    const int* tileCounts;
    const int* tileValues;
    const uint64_t* tileKeys;

    const float2* positionClipXY;
    const float4* conic; // (A,B,C,0)
    const float4* color; // (r,g,b,opacity)
};

__device__ __forceinline__ void swap_u64(uint64_t& a, uint64_t& b)
{
    uint64_t t = a; a = b; b = t;
}

__device__ __forceinline__ void swap_i32(int& a, int& b)
{
    int t = a; a = b; b = t;
}

// Very small shared-memory sort for 32 items (bitonic), used per chunk.
__device__ __forceinline__ void bitonic_sort_32(uint64_t* keys, int* values)
{
    // Assumes all threads in the block participate, but only first 32 lanes touch.
    int lane = (int)threadIdx.x;
    if (lane >= 32) return;

    for (int k = 2; k <= 32; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            int ixj = lane ^ j;
            if (ixj > lane)
            {
                bool up = ((lane & k) == 0);
                uint64_t keyA = keys[lane];
                uint64_t keyB = keys[ixj];
                if ((keyA > keyB) == up)
                {
                    keys[lane] = keyB;
                    keys[ixj] = keyA;
                    int va = values[lane];
                    values[lane] = values[ixj];
                    values[ixj] = va;
                }
            }
            __syncwarp();
        }
    }
}

__global__ void gsim_raster_kernel(RasterIn in, uchar4* outBackBuffer)
{
    const int tileId = (int)blockIdx.x;

    // Pixel within tile.
    const int pixInTile = (int)threadIdx.x;
    const int px = pixInTile % (int)GSIM_TILE_SIZE;
    const int py = pixInTile / (int)GSIM_TILE_SIZE;

    const int tileX = tileId % (int)GSIM_TILES_PER_SCREEN;
    const int tileY = tileId / (int)GSIM_TILES_PER_SCREEN;

    const int screenX = tileX * (int)GSIM_TILE_SIZE + px;
    const int screenY = tileY * (int)GSIM_TILE_SIZE + py;

    const float2 clip = make_float2(
        (float)screenX * (2.0f / (float)GSIM_SCREEN_SIZE) - 1.0f,
        (float)screenY * (2.0f / (float)GSIM_SCREEN_SIZE) - 1.0f);

    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    float T = 1.0f;

    const int count = in.tileCounts[tileId];
    const int maxCount = count < (int)GSIM_MAX_SPLATS_PER_TILE ? count : (int)GSIM_MAX_SPLATS_PER_TILE;

    __shared__ int s_splatIds[32];
    __shared__ uint64_t s_keys[32];
    __shared__ float2 s_centers[32];
    __shared__ float4 s_conics[32];
    __shared__ float4 s_colors[32];

    for (int base = 0; base < maxCount; base += 32)
    {
        int lane = (int)threadIdx.x;
        if (lane < 32)
        {
            int idx = base + lane;
            if (idx < maxCount)
            {
                int splatId = in.tileValues[tileId * (int)GSIM_MAX_SPLATS_PER_TILE + idx];
                uint64_t key = in.tileKeys[tileId * (int)GSIM_MAX_SPLATS_PER_TILE + idx];
                s_splatIds[lane] = splatId;
                s_keys[lane] = key;
            }
            else
            {
                s_splatIds[lane] = -1;
                s_keys[lane] = 0xFFFFFFFFFFFFFFFFull;
            }
        }

        __syncthreads();

        // Sort the chunk front-to-back by depth key (ascending).
        bitonic_sort_32(s_keys, s_splatIds);

        __syncthreads();

        // Load attributes for the sorted chunk.
        if (threadIdx.x < 32)
        {
            int splatId = s_splatIds[threadIdx.x];
            if (splatId >= 0)
            {
                s_centers[threadIdx.x] = in.positionClipXY[splatId];
                s_conics[threadIdx.x] = in.conic[splatId];
                s_colors[threadIdx.x] = in.color[splatId];
            }
        }

        __syncthreads();

        // Blend this chunk.
        for (int i = 0; i < 32; ++i)
        {
            int splatId = s_splatIds[i];
            if (splatId < 0) break;

            float2 d = f2_sub(clip, s_centers[i]);
            float4 con = s_conics[i];
            float4 col = s_colors[i];

            float dx = con.x * d.x * d.x + con.z * d.y * d.y + 2.0f * con.y * d.x * d.y;
            float density = __expf(-0.5f * dx);
            float a = col.w * __saturatef(density);

            accum.x += col.x * T * a;
            accum.y += col.y * T * a;
            accum.z += col.z * T * a;
            T *= (1.0f - a);

            if (T < 0.02f) break;
        }

        __syncthreads();

        // Early out if the whole block is done.
        if (__syncthreads_count(T > 0.02f) == 0) break;
    }

    uchar4 out; 
    out.x = (unsigned char)(clampf(accum.x, 0.0f, 1.0f) * 255.0f);
    out.y = (unsigned char)(clampf(accum.y, 0.0f, 1.0f) * 255.0f);
    out.z = (unsigned char)(clampf(accum.z, 0.0f, 1.0f) * 255.0f);
    out.w = 255;

    outBackBuffer[screenY * (int)GSIM_SCREEN_SIZE + screenX] = out;
}

extern "C" void gsim_rasterize(
    const int* d_tileCounts,
    const int* d_tileValues,
    const uint64_t* d_tileKeys,
    const float2* d_positionClipXY,
    const float4* d_conic,
    const float4* d_color,
    uchar4* d_backBuffer)
{
    RasterIn in;
    in.tileCounts = d_tileCounts;
    in.tileValues = d_tileValues;
    in.tileKeys = d_tileKeys;
    in.positionClipXY = d_positionClipXY;
    in.conic = d_conic;
    in.color = d_color;

    const dim3 block((unsigned)(GSIM_TILE_SIZE * GSIM_TILE_SIZE));
    const dim3 grid((unsigned)GSIM_TOTAL_TILES);

    gsim_raster_kernel<<<grid, block>>>(in, d_backBuffer);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("gsim_raster kernel error: %s\n", cudaGetErrorString(err));
    }
}
