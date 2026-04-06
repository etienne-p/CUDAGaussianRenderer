
#define CHECK_CUDA_ERRORS(call)                                                               \
    do                                                                                        \
    {                                                                                         \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess)                                                               \
        {                                                                                     \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)
