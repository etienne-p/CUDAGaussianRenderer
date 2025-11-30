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

// Used to generate random test splats.
void generateRandomGaussians(
    std::vector<float4>& positionWorldSpace,
    std::vector<float4>& covarianceWorldSpace,
    const float minScale,
    const float maxScale,
    const glm::vec4& minPosition,
    const glm::vec4& maxPosition)
{
    for (auto i = 0; i != positionWorldSpace.size(); ++i)
    {
        auto rotAxis = glm::sphericalRand(1.0f);
        auto rotAngle = glm::linearRand(0.0f, glm::pi<float>());
        auto rotation = glm::mat3_cast(glm::angleAxis(rotAngle, rotAxis));
        auto scaleTrace =
            glm::linearRand(glm::vec3(minScale, minScale, minScale), glm::vec3(maxScale, maxScale, maxScale));
        auto scale = glm::mat3(0);
        scale[0][0] = scaleTrace[0];
        scale[1][1] = scaleTrace[1];
        scale[2][2] = scaleTrace[2];

        auto RS = rotation * scale;
        auto cov = RS * glm::transpose(RS);
        // Covariance is symmetric.
        covarianceWorldSpace[i * 2 + 0] = float4{cov[0][0], cov[1][0], cov[2][0], 0};
        covarianceWorldSpace[i * 2 + 1] = float4{cov[1][1], cov[2][1], cov[2][2], 0};
        positionWorldSpace[i] = glmToBuiltin4(glm::linearRand(minPosition, maxPosition));
    }
}

constexpr uint32_t k_QuadIndices[] = {0, 1, 2, 2, 3, 0};
const glm::vec3 k_QuadVertices[] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
const glm::vec2 k_QuadUvs[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

struct Stats
{
    double evaluateClipData{0};
    double buildTileList{0};
    double sortTileList{0};
    double evaluateTileRanges{0};
    double renderDepthBuffer{0};
};

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

#if false
    // Use randomly generated splats.
    constexpr int32_t splatCount = 1 << 4;
    constexpr auto worldBoundsExtent = 4.0f;
    cameraControls.setBounds(glm::vec3(-worldBoundsExtent), glm::vec3(worldBoundsExtent))

    const auto minPosition = glm::vec4(-worldBoundsExtent, -worldBoundsExtent, -worldBoundsExtent, 1.0f);
    const auto maxPosition = glm::vec4(worldBoundsExtent, worldBoundsExtent, worldBoundsExtent, 1.0f);

    auto positionWorldSpace = std::vector<float4>(splatCount);
    auto covarianceWorldSpace = std::vector<float4>(splatCount * 2);
    generateRandomGaussians(
        positionWorldSpace,
        covarianceWorldSpace,
        0.01f,
        0.5f,
        minPosition,
        maxPosition);

    auto color = std::vector<float4>(splatCount);
    for (auto i = 0; i != splatCount; ++i)
    {
        color[i] = glmToBuiltin4(glm::linearRand(glm::vec4(0), glm::vec4(1)));
    }

#else
    // Load splats from a .ply file.
    std::vector<float4> positionWorldSpace;
    std::vector<float4> covarianceWorldSpace;
    std::vector<float4> color;

    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    auto splatCount = ParsePly(argv[1], positionWorldSpace, covarianceWorldSpace, color, boundsMin, boundsMax);

    // Orbit around dataset, based on the dataset bounding box.
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
    auto positionWorldSpaceBuffer = DeviceBuffer<float4>(positionWorldSpace);
    auto covarianceWorldSpaceBuffer = DeviceBuffer<float4>(covarianceWorldSpace);
    auto colorBuffer = DeviceBuffer<float4>(color);
    auto conicBuffer = DeviceBuffer<float4>(splatCount);
    auto positionClipSpaceXYBuffer = DeviceBuffer<float2>(splatCount);
    auto positionClipSpaceZBuffer = DeviceBuffer<float>(splatCount);
    auto screenEllipseBuffer = DeviceBuffer<float4>(splatCount);
    auto tileRangeBuffer = DeviceBuffer<int32_t>(k_TotalTiles * 2);

    // Static size, no dynamic alloc for now.
    auto tileListCapacity = splatCount * 256;

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
    long frameCount = 0;
    auto cudaTimer = CudaTimer();

    // We use a timer to cap frame rate.
    auto lastTime = glfwGetTime();
    auto deltaTime = 0.0f;

    // Update loop.
    do
    {
        ++frameCount;

        // Update time.
        auto time = glfwGetTime();
        deltaTime = time - lastTime;
        lastTime = time;

        // Update input & camera controls.
        cameraControls.update((float) deltaTime);

        // Build camera data.
        CameraData cameraData;
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

            // Set global gata.
            GlobalArgs globalArgs;
            globalArgs.splatCount = splatCount;
            globalArgs.positionClipSpaceXY = positionClipSpaceXYBuffer.getPtr();
            globalArgs.positionClipSpaceZ = positionClipSpaceZBuffer.getPtr();
            globalArgs.screenEllipse = screenEllipseBuffer.getPtr();
            globalArgs.positionWorldSpace = positionWorldSpaceBuffer.getPtr();
            globalArgs.covarianceWorldSpace = covarianceWorldSpaceBuffer.getPtr();
            globalArgs.color = colorBuffer.getPtr();
            globalArgs.conic = conicBuffer.getPtr();
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

            evaluateSplatClipData(cudaTimer, splatCount);
            stats.evaluateClipData += (double) cudaTimer.getElapseTimedMs();

            // Multiple blocks per multiprocessor, to give the scheduler a chance to hide memory latency.
            tileListArgs.size = buildTileList(cudaTimer, prop.multiProcessorCount * 8, tileListCapacity);
            assert(tileListArgs.size <= tileListCapacity);
            stats.buildTileList += (double) cudaTimer.getElapseTimedMs();

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

    printf("evaluateClipData average time ms: %2.6f\n", stats.evaluateClipData);
    printf("buildTileList average time ms: %2.6f\n", stats.buildTileList);
    printf("sortTileList average time ms: %2.6f\n", stats.sortTileList);
    printf("evaluateTileRanges average time ms: %2.6f\n", stats.evaluateTileRanges);
    printf("renderDepthBuffer average time ms: %2.6f\n", stats.renderDepthBuffer);
    printf("Total average time ms: %2.6f\n", totalMs);
    getchar();

    return 0;
}
