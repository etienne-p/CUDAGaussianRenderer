#define GLM_FORCE_SWIZZLE

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
GLFWwindow* window;

#include "Consts.h"
#include "cuda_gl_interop.h"
#include "DeviceBuffer.cu"
#include "GaussianRender.cuh"
#include "PlyParser.h"
#include "CameraControls.h"

#define STRINGIFY(A) #A

// clang-format off
// Shaders used to display the texture on screen.
const char* vertexShaderTextured = STRINGIFY(
    #version 330 core \n
    layout(location = 0) in vec3 vertexPosition_modelspace; \n 
    layout(location = 1) in vec2 vertexUv; \n 
    out vec2 uv; \n 
    void main() \n 
    { \n 
        gl_Position = vec4(vertexPosition_modelspace, 1); \n
        uv = vertexUv; \n
    } \n
);

const char* fragmentShaderTextured = STRINGIFY(
    #version 330 core \n
    in vec2 uv; \n 
    out vec4 color; \n 
    uniform sampler2D textureSampler; \n
    void main() \n 
    { \n 
        color = texture(textureSampler, uv).rgba; \n
    } \n
);
// clang-format on

GLuint compileShadersProgram(const char* vertexShader, const char* fragmentShader)
{
    // Create the shaders
    auto vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    auto fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);

    GLint result = GL_FALSE;
    int infoLogLength;

    // Compile Vertex Shader
    glShaderSource(vertexShaderId, 1, &vertexShader, NULL);
    glCompileShader(vertexShaderId);

    // Check Vertex Shader
    glGetShaderiv(vertexShaderId, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertexShaderId, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        std::vector<char> VertexShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(vertexShaderId, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }

    // Compile Fragment Shader
    glShaderSource(fragmentShaderId, 1, &fragmentShader, NULL);
    glCompileShader(fragmentShaderId);

    // Check Fragment Shader
    glGetShaderiv(fragmentShaderId, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragmentShaderId, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        std::vector<char> FragmentShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(fragmentShaderId, infoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    // Link the program
    auto programId = glCreateProgram();
    glAttachShader(programId, vertexShaderId);
    glAttachShader(programId, fragmentShaderId);
    glLinkProgram(programId);

    // Check the program
    glGetProgramiv(programId, GL_LINK_STATUS, &result);
    glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        std::vector<char> ProgramErrorMessage(infoLogLength + 1);
        glGetProgramInfoLog(programId, infoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }

    glDetachShader(programId, vertexShaderId);
    glDetachShader(programId, fragmentShaderId);

    glDeleteShader(vertexShaderId);
    glDeleteShader(fragmentShaderId);

    return programId;
}

float4 glmToBuiltin4(const glm::vec4& v)
{
    return float4{v.x, v.y, v.z, v.w};
}

// TODO: Add color buffer.
// Used to generate random test splats.
void generateRandomGaussians(
    std::vector<float4>& position,
    std::vector<float4>& scaleAndRotation,
    std::vector<float4>& color,
    const float minScale,
    const float maxScale,
    const glm::vec4& minPosition,
    const glm::vec4& maxPosition)
{
    for (auto i = 0; i != position.size(); ++i)
    {
        auto translation = glm::linearRand(minPosition, maxPosition);
        auto rotAxis = glm::sphericalRand(1.0f);
        auto rotAngle = glm::linearRand(0.0f, glm::pi<float>());
        auto rotation = glm::angleAxis(rotAngle, rotAxis);
        auto scale = glm::linearRand(glm::vec3(minScale, minScale, minScale), glm::vec3(maxScale, maxScale, maxScale));

        auto col = glm::linearRand(glm::vec4(0), glm::vec4(1));
        auto quantizedRotation = encodeVec4((glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w) + 1.0f) * 0.5f);

        position[i] = float4{translation.x, translation.y, translation.z, 1.0f};
        scaleAndRotation[i] = float4{scale.x, scale.y, scale.z, reinterpret_cast<float&>(quantizedRotation)};
        color[i] = float4{col.x, col.y, col.z, col.w};
    }
}

constexpr uint32_t k_QuadIndices[] = {0, 1, 2, 2, 3, 0};
const glm::vec3 k_QuadVertices[] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
const glm::vec2 k_QuadUvs[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

struct Stats
{
    double evaluateSphericalHarmonics{0};
    double evaluateClipData{0};
    double buildTileList{0};
    double sortTileList{0};
    double evaluateTileRanges{0};
    double renderDepthBuffer{0};
};

// Re-align spherical harmonics for more efficient memory usage on the GPU.
void realignSphericalHarmonics(
    const std::vector<float>& srcSh,
    std::vector<float>& dstSh,
    const int groupSize,
    const int shCount,
    const int splatCount)
{
    auto shCountPerComponent = shCount / 3;
    // Verify it's a multiple.
    assert(shCountPerComponent * 3 == shCount);

    auto idx = 0;
    auto groupCount = (int) glm::ceil(splatCount / (float) groupSize);

    // We may have unused space for the last group but we do not break alignment.
    dstSh.resize(groupCount * groupSize * shCount);

    for (auto grp = 0; grp != groupCount; ++grp)
    {
        auto start = grp * groupSize * shCount;
        auto thisGroupSize = glm::min(groupSize, splatCount - grp * groupSize);

        // For all spherical harmonics.
        for (auto i = 0; i != shCount; ++i)
        {
            // For each group item.
            for (auto k = 0; k != thisGroupSize; ++k)
            {
                auto srcIdx = shCount * k + i;
                auto dstIdx = groupSize * i + k;
                dstSh[start + dstIdx] = srcSh[start + srcIdx];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                   GL_TRUE); // To make macOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(k_ScreenSize, k_ScreenSize, "GaussianRenderer", NULL, NULL);
    if (window == NULL)
    {
        fprintf(stderr,
                "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 "
                "compatible. Try the 2.1 version of the tutorials.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Device selection
    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    checkCudaErrors(cudaChooseDevice(&dev, &prop));
    checkCudaErrors(cudaGLSetGLDevice(dev));

    cudaGetDeviceProperties(&prop, dev);

    auto cameraControls = CameraControls(window, glm::vec2(k_ScreenSize));

    std::vector<float> sphericalHarmonics;
    int sphericalHarmonicsDegree{0};
    int sphericalHarmonicsCount{0};

#if false
    // Use randomly generated splats. Useful for testing and debugging.
    constexpr int32_t splatCount = 1 << 4;
    constexpr auto worldBoundsExtent = 4.0f;

    const auto minPosition = glm::vec4(-worldBoundsExtent, -worldBoundsExtent, -worldBoundsExtent, 1.0f);
    const auto maxPosition = glm::vec4(worldBoundsExtent, worldBoundsExtent, worldBoundsExtent, 1.0f);

    auto position = std::vector<float4>(splatCount);
    auto scaleAndRotation = std::vector<float4>(splatCount);
    auto color = std::vector<float4>(splatCount);
    generateRandomGaussians(position, scaleAndRotation, color, 0.01f, 0.5f, minPosition, maxPosition);

    cameraControls.setBounds(glm::vec3(-worldBoundsExtent), glm::vec3(worldBoundsExtent));

#else
    // Load splats from a .ply file.
    std::vector<float4> position;
    std::vector<float4> scaleAndRotation;
    std::vector<float4> color;
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    auto splatCount = parsePly(
        argv[1],
        position,
        scaleAndRotation,
        color,
        sphericalHarmonics,
        sphericalHarmonicsDegree,
        sphericalHarmonicsCount,
        boundsMin,
        boundsMax);
    auto hasExtraSphericalHarmonics = sphericalHarmonicsDegree > 1;

    std::vector<float> alignedSphericalHarmonics;
    if (sphericalHarmonicsDegree != 0)
    {
        realignSphericalHarmonics(
            sphericalHarmonics, alignedSphericalHarmonics, 256, sphericalHarmonicsCount, splatCount);
        std::swap(sphericalHarmonics, alignedSphericalHarmonics);
    }

    cameraControls.setBounds(boundsMin, boundsMax);

#endif

    // Create and compile the GLSL program. Used to draw the texture CUDA renders to.
    auto programTexturedId = compileShadersProgram(vertexShaderTextured, fragmentShaderTextured);
    auto textureId = glGetUniformLocation(programTexturedId, "textureSampler");

    auto vertexArray = GLVertexArray();
    // OpenGL buffers.
    auto glVertexBuffer = GLBuffer<glm::vec3, GL_ARRAY_BUFFER>(4, k_QuadVertices);
    auto glUvBuffer = GLBuffer<glm::vec2, GL_ARRAY_BUFFER>(4, k_QuadUvs);
    auto glIndexBuffer = GLBuffer<uint32_t, GL_ARRAY_BUFFER>(6, k_QuadIndices);
    auto glColorBuffer = GLBuffer<uchar4, GL_PIXEL_UNPACK_BUFFER>(k_ScreenSize * k_ScreenSize);

    // CUDA buffers.
    auto positionBuffer = DeviceBuffer<float4>(position);
    auto scaleAndRotationBuffer = DeviceBuffer<float4>(scaleAndRotation);
    auto colorBuffer = DeviceBuffer<float4>(color);
    auto sphericalHarmonicsBuffer = DeviceBuffer<float>(sphericalHarmonics);
    auto conicAndColorBuffer = DeviceBuffer<float4>(splatCount);
    auto positionClipSpaceXYBuffer = DeviceBuffer<float2>(splatCount);
    auto positionClipSpaceZBuffer = DeviceBuffer<float>(splatCount);
    auto screenEllipseBuffer = DeviceBuffer<float4>(splatCount);
    auto tileRangeBuffer = DeviceBuffer<int32_t>(k_TotalTiles * 2);

    // Static size, no dynamic alloc for now.
    auto tileListCapacity = splatCount * 8;

    // Double buffers for the tile list.
    auto tileListKeysCurrentBuffer = DeviceBuffer<uint64_t>(tileListCapacity);
    auto tileListValuesCurrentBuffer = DeviceBuffer<int32_t>(tileListCapacity);
    auto tileListKeysAlternateBuffer = DeviceBuffer<uint64_t>(tileListCapacity);
    auto tileListValuesAlternateBuffer = DeviceBuffer<int32_t>(tileListCapacity);

    // Radix sort temporary memory;
    void* deviceTempStorage{nullptr};
    size_t tempStorageSizeInBytes{0};

    // Share OpenGL color buffer with CUDA. It's the buffer we render to.
    auto colorBufferResource = CudaGraphicsResource(glColorBuffer.getBufferId(), cudaGraphicsRegisterFlagsNone);

    Stats stats;
    long frameCount{0};
    auto cudaTimer = CudaTimer();

    // We use a timer to cap frame rate.
    auto lastTime = glfwGetTime();
    auto deltaTime{0.0f};

    // Track whether the tile list is saturated.
    auto tileListIsSaturated{false};

    // Update loop.
    do
    {
        ++frameCount;

        // Re-allocate tile list if needed.
        if (tileListIsSaturated)
        {
            // Double the size of the tile list.
            tileListCapacity <<= 1;
            tileListKeysCurrentBuffer.resizeIfNeeded(tileListCapacity);
            tileListValuesCurrentBuffer.resizeIfNeeded(tileListCapacity);
            tileListKeysAlternateBuffer.resizeIfNeeded(tileListCapacity);
            tileListValuesAlternateBuffer.resizeIfNeeded(tileListCapacity);
            tileListIsSaturated = false;
        }

        // Update time.
        auto time = glfwGetTime();
        deltaTime = time - lastTime;
        lastTime = time;

        // Update input & camera controls.
        cameraControls.update((float) deltaTime);

        // Build camera data.
        CameraData cameraData;
        cameraData.position = cameraControls.getPosition();
        cameraData.aspect = cameraControls.getAspect();
        cameraData.projection = cameraControls.getProjection();
        cameraData.viewProjection = cameraControls.getViewProjection();
        cameraData.view = cameraControls.getView();
        auto cotangentY = 1.0f / glm::tan(cameraControls.getFieldOfView() * 0.5f);
        auto cotangentX = cotangentY / cameraControls.getAspect();
        cameraData.fovCotangent = glm::vec2(cotangentX, cotangentY);
        // Orthographic mapping of the Z axis.
        // We use the default right handed coordinates, so we flip Z via scaleZ.
        // Clip depth is in [0, 1], this range matters when evaluating sorting keys.
        auto scaleZ = -2.0f / (cameraControls.getFar() - cameraControls.getNear());
        auto translationZ = -(cameraControls.getFar() + cameraControls.getNear())
                          / (cameraControls.getFar() - cameraControls.getNear());
        cameraData.depthScaleBias = glm::vec2(scaleZ, translationZ);

        // CUDA Update.
        {
            auto colorBinding = colorBufferResource.getBinding<uchar4>();

            // Clear color buffer.
            checkCudaErrors(cudaMemset(colorBinding.getPtr(), 0, glColorBuffer.getSizeInBytes()));

            // Clear tile ranges. Needed to identify empty ranges.
            // Set all pointers to 0xFFFFFFFF = -1
            tileRangeBuffer.clearMemory(255);

            // Set global data.
            GlobalArgs globalArgs;
            globalArgs.splatCount = splatCount;
            globalArgs.positionClipSpaceXY = positionClipSpaceXYBuffer.getPtr();
            globalArgs.positionClipSpaceZ = positionClipSpaceZBuffer.getPtr();
            globalArgs.sphericalHarmonicsDegree = sphericalHarmonicsDegree;
            globalArgs.sphericalHarmonicsCount = sphericalHarmonicsCount;
            globalArgs.sphericalHarmonics = sphericalHarmonicsBuffer.getPtr();
            globalArgs.screenEllipse = screenEllipseBuffer.getPtr();
            globalArgs.position = positionBuffer.getPtr();
            globalArgs.scaleAndRotation = scaleAndRotationBuffer.getPtr();
            globalArgs.color = colorBuffer.getPtr();
            globalArgs.conic = conicAndColorBuffer.getPtr();
            globalArgs.tileRange = tileRangeBuffer.getPtr();
            globalArgs.backBuffer = colorBinding.getPtr();
            globalArgs.cameraData = cameraData;

            setGlobalArgs(&globalArgs);

            // Pass current tile list buffers.
            TileListArgs tileListArgs;
            tileListArgs.keys = tileListKeysCurrentBuffer.getPtr();
            tileListArgs.values = tileListValuesCurrentBuffer.getPtr();
            tileListArgs.capacity = tileListCapacity;

            setTileListArgs(&tileListArgs);

            if (sphericalHarmonicsDegree != 0)
            {
                evaluateSphericalHarmonics(cudaTimer, splatCount);
                stats.evaluateSphericalHarmonics += (double) cudaTimer.getElapseTimedMs();
            }

            evaluateSplatClipData(cudaTimer, splatCount);
            stats.evaluateClipData += (double) cudaTimer.getElapseTimedMs();

            // Multiple blocks per multiprocessor, to give the scheduler a chance to hide memory latency.
            tileListArgs.size = buildTileList(cudaTimer, prop.multiProcessorCount * 8, tileListCapacity);
            assert(tileListArgs.size <= tileListCapacity);
            stats.buildTileList += (double) cudaTimer.getElapseTimedMs();

            // Track whether the tile list is saturated.
            tileListIsSaturated = tileListArgs.size == tileListCapacity;

            // Update render list args with render list size.
            setTileListArgs(&tileListArgs);

            // Instantiate double buffers.
            DoubleBuffer<uint64_t> tileListKeys(
                tileListKeysCurrentBuffer.getPtr(), tileListKeysAlternateBuffer.getPtr());
            DoubleBuffer<int32_t> tileListValues(
                tileListValuesCurrentBuffer.getPtr(), tileListValuesAlternateBuffer.getPtr());

            sortTileList(
                cudaTimer, tileListArgs.size, deviceTempStorage, tempStorageSizeInBytes, tileListKeys, tileListValues);
            stats.sortTileList += (double) cudaTimer.getElapseTimedMs();

            // Update render list args with current buffers.
            tileListArgs.keys = tileListKeys.current();
            tileListArgs.values = tileListValues.current();

            setTileListArgs(&tileListArgs);

            // Bypass if no splat is visible.
            if (tileListArgs.size != 0)
            {
                evaluateTileRange(cudaTimer, tileListArgs.size);
                stats.evaluateTileRanges += (double) cudaTimer.getElapseTimedMs();
            }

            rasterizeTile(cudaTimer);
            stats.renderDepthBuffer += (double) cudaTimer.getElapseTimedMs();

            checkCudaErrors(cudaDeviceSynchronize());
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        // Draw the rendered texture.
        {
            glUseProgram(programTexturedId);
            glEnable(GL_TEXTURE_2D);

            auto vertexBinding = glVertexBuffer.getBinding(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
            auto uvBinding = glUvBuffer.getBinding(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*) 0);

            glActiveTexture(GL_TEXTURE0);
            // This code is using the immediate mode texture object 0.
            glBindTexture(GL_TEXTURE_2D, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // From https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
            // If a non-zero named buffer object is bound to the GL_PIXEL_UNPACK_BUFFER target (see
            // glBindBuffer) while a texture image is specified, data is treated as a byte offset
            // into the buffer object's data store.
            auto depthBinding = glColorBuffer.getBinding();
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, k_ScreenSize, k_ScreenSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

            glUniform1i(textureId, 0);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIndexBuffer.getBufferId());
            glDrawElements(GL_TRIANGLES, glIndexBuffer.getSize(), GL_UNSIGNED_INT, (void*) 0);

            glDisable(GL_TEXTURE_2D);
        }

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Limit framerate.
        while (glfwGetTime() < lastTime + 1.0 / 60.0)
        {
            // Nothing, wait.
        }

    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

    // Release radix sort temporary memory.
    if (deviceTempStorage != nullptr)
    {
        checkCudaErrors(cudaFree(deviceTempStorage));
    }

    glDeleteProgram(programTexturedId);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    // Evaluate profiling averages and print to terminal.
    stats.evaluateSphericalHarmonics /= (double) frameCount;
    stats.evaluateClipData /= (double) frameCount;
    stats.buildTileList /= (double) frameCount;
    stats.sortTileList /= (double) frameCount;
    stats.evaluateTileRanges /= (double) frameCount;
    stats.renderDepthBuffer /= (double) frameCount;

    auto totalMs = //
        stats.evaluateClipData + //
        stats.buildTileList + //
        stats.sortTileList + //
        stats.evaluateTileRanges + //
        stats.renderDepthBuffer; //

    printf("evaluateSphericalHarmonics average time ms: %2.6f\n", stats.evaluateSphericalHarmonics);
    printf("evaluateClipData average time ms: %2.6f\n", stats.evaluateClipData);
    printf("buildTileList average time ms: %2.6f\n", stats.buildTileList);
    printf("sortTileList average time ms: %2.6f\n", stats.sortTileList);
    printf("evaluateTileRanges average time ms: %2.6f\n", stats.evaluateTileRanges);
    printf("renderDepthBuffer average time ms: %2.6f\n", stats.renderDepthBuffer);
    printf("Total average time ms: %2.6f\n", totalMs);
    getchar();

    return 0;
}
