#include "common.cuh"
#include <stdio.h>

// Screen-space preprocess:
// - Takes per-splat clip-space center (XY), clip-space depth (Z), and a 2D covariance (a,b,c).
// - Outputs ellipse parameters (cos/sin, extent) and conic coefficients (inverse covariance).
// This intentionally isolates the math-heavy part of the pipeline without depending on 3D transforms.

struct PreprocessOut
{
    float2* positionClipXY;
    float* positionClipZ;
    float4* screenEllipse; // (cos, sin, extentX, extentY)
    float4* conic;         // (A,B,C,0) for symmetric matrix [[A,B],[B,C]]
};

__global__ void gsim_preprocess_kernel(
    int splatCount,
    const float2* inClipXY,
    const float* inClipZ,
    const Cov2* inCov,
    PreprocessOut out,
    float sigmaScale)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= splatCount) return;

    float2 c = inClipXY[i];
    float z = inClipZ[i];
    Cov2 cov = inCov[i];

    // Basic frustum reject on center; keep it cheap.
    bool inFrustum = (c.x >= -1.0f && c.x <= 1.0f && c.y >= -1.0f && c.y <= 1.0f && z >= -1.0f && z <= 1.0f);

    // Eigenvalues of 2x2 symmetric matrix.
    float det = cov.a * cov.c - cov.b * cov.b;
    float mid = 0.5f * (cov.a + cov.c);
    float disc = mid * mid - det;
    disc = disc < 0.0f ? 0.0f : disc;
    float radius = sqrtf(disc);
    float l0 = mid + radius;
    float l1 = mid - radius;

    // Clamp to avoid NaNs.
    const float eps = 1e-12f;
    l0 = l0 < eps ? eps : l0;
    l1 = l1 < eps ? eps : l1;

    // Eigenvector for l0: [b, l0-a]
    float vx = cov.b;
    float vy = (l0 - cov.a);
    float vlen = sqrtf(vx * vx + vy * vy);
    if (vlen < eps) { vx = 1.0f; vy = 0.0f; vlen = 1.0f; }
    vx /= vlen;
    vy /= vlen;

    float angleCos = vx;
    float angleSin = vy;

    float ex = sigmaScale * sqrtf(l0);
    float ey = sigmaScale * sqrtf(l1);

    // Inverse covariance (conic).
    float invDet = 1.0f / (fabsf(det) < eps ? eps : det);
    float A = cov.c * invDet;
    float B = -cov.b * invDet;
    float C = cov.a * invDet;

    if (!inFrustum)
    {
        // Mark as invalid / invisible.
        c = make_float2(-128.0f, -128.0f);
        ex = 0.0f;
        ey = 0.0f;
        A = 0.0f;
        B = 0.0f;
        C = 0.0f;
    }

    out.positionClipXY[i] = c;
    out.positionClipZ[i] = z;
    out.screenEllipse[i] = make_float4(angleCos, angleSin, ex, ey);
    out.conic[i] = make_float4(A, B, C, 0.0f);
}

extern "C" void gsim_preprocess(
    int splatCount,
    const float2* d_inClipXY,
    const float* d_inClipZ,
    const Cov2* d_inCov,
    float2* d_outClipXY,
    float* d_outClipZ,
    float4* d_outScreenEllipse,
    float4* d_outConic,
    float sigmaScale)
{
    PreprocessOut out;
    out.positionClipXY = d_outClipXY;
    out.positionClipZ = d_outClipZ;
    out.screenEllipse = d_outScreenEllipse;
    out.conic = d_outConic;

    const int threads = 256;
    const int blocks = (splatCount + threads - 1) / threads;
    gsim_preprocess_kernel<<<blocks, threads>>>(splatCount, d_inClipXY, d_inClipZ, d_inCov, out, sigmaScale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("gsim_preprocess kernel error: %s\n", cudaGetErrorString(err));
    }
}
