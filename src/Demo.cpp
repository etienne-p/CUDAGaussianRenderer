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

// clang-format off
// Shaders used to display the texture on screen.
const char* kVertexShaderTextured = R"(
    #version 330 core
    layout(location = 0) in vec3 vertexPosition_modelspace;
    layout(location = 1) in vec2 vertexUv;
    out vec2 uv;
    void main()
    {
        gl_Position = vec4(vertexPosition_modelspace, 1);
        uv = vertexUv;
    }
)";

const char* kFragmentShaderTextured = R"(
    #version 330 core
    in vec2 uv;
    out vec4 color;
    uniform sampler2D textureSampler;
    void main()
    {
        color = texture(textureSampler, uv).rgba;
    }
)";
// clang-format on

GLuint CompileShadersProgram(const char* vertex_shader, const char* fragment_shader)
{
    // Create the shaders
    auto vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    auto fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);

    GLint result = GL_FALSE;
    int info_log_length;

    // Compile Vertex Shader
    glShaderSource(vertex_shader_id, 1, &vertex_shader, NULL);
    glCompileShader(vertex_shader_id);

    // Check Vertex Shader
    glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertex_shader_id, GL_INFO_LOG_LENGTH, &info_log_length);

    if (info_log_length > 0)
    {
        std::vector<char> vertex_shader_error_message(info_log_length + 1);
        glGetShaderInfoLog(vertex_shader_id, info_log_length, NULL, &vertex_shader_error_message[0]);
        printf("%s\n", &vertex_shader_error_message[0]);
    }

    // Compile Fragment Shader
    glShaderSource(fragment_shader_id, 1, &fragment_shader, NULL);
    glCompileShader(fragment_shader_id);

    // Check Fragment Shader
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragment_shader_id, GL_INFO_LOG_LENGTH, &info_log_length);

    if (info_log_length > 0)
    {
        std::vector<char> fragment_shader_error_message(info_log_length + 1);
        glGetShaderInfoLog(fragment_shader_id, info_log_length, NULL, &fragment_shader_error_message[0]);
        printf("%s\n", &fragment_shader_error_message[0]);
    }

    // Link the program
    auto program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader_id);
    glAttachShader(program_id, fragment_shader_id);
    glLinkProgram(program_id);

    // Check the program
    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);

    if (info_log_length > 0)
    {
        std::vector<char> program_error_message(info_log_length + 1);
        glGetProgramInfoLog(program_id, info_log_length, NULL, &program_error_message[0]);
        printf("%s\n", &program_error_message[0]);
    }

    glDetachShader(program_id, vertex_shader_id);
    glDetachShader(program_id, fragment_shader_id);

    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    return program_id;
}

float4 GlmToBuiltin4(const glm::vec4& v)
{
    return float4{v.x, v.y, v.z, v.w};
}

// TODO: Add color buffer.
// Used to generate random test splats.
void GenerateRandomGaussians(
    std::vector<float4>& positions,
    std::vector<float4>& scales,
    std::vector<float4>& rotations,
    std::vector<float4>& colors,
    const float scaleMin,
    const float scaleMax,
    const glm::vec4& boundsMin,
    const glm::vec4& boundsMax)
{
    for (auto i = 0; i != positions.size(); ++i)
    {
        auto translation = glm::linearRand(boundsMin, boundsMax);
        auto rotAxis = glm::sphericalRand(1.0f);
        auto rotAngle = glm::linearRand(0.0f, glm::pi<float>());
        auto rotation = glm::angleAxis(rotAngle, rotAxis);
        auto scale = glm::linearRand(glm::vec3(scaleMin, scaleMin, scaleMin), glm::vec3(scaleMax, scaleMax, scaleMax));

        auto col = glm::linearRand(glm::vec4(0), glm::vec4(1));

        positions[i] = float4{translation.x, translation.y, translation.z, 1.0f};
        scales[i] = float4{scale.x, scale.y, scale.z, 0.f};
        rotations[i] = float4{rotation.x, rotation.y, rotation.z, rotation.w};
        colors[i] = float4{col.x, col.y, col.z, col.w};
    }
}

constexpr uint32_t kQuadIndices[] = {0, 1, 2, 2, 3, 0};
const glm::vec3 kQuadVertices[] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
const glm::vec2 kQuadUvs[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

struct Stats
{
    double evaluate_spherical_harmonics{0};
    double evaluate_clip_data{0};
    double build_tile_list{0};
    double sort_tile_list{0};
    double evaluate_tile_ranges{0};
    double render_depth_buffer{0};
};

// Re-align spherical harmonics for more efficient memory usage on the GPU.
void RealignSphericalHarmonics(
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
    window = glfwCreateWindow(kScreenSize, kScreenSize, "GaussianRenderer", NULL, NULL);
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
    CHECK_CUDA_ERRORS(cudaChooseDevice(&dev, &prop));
    CHECK_CUDA_ERRORS(cudaGLSetGLDevice(dev));

    cudaGetDeviceProperties(&prop, dev);

    auto camera_controls = CameraControls(window, glm::vec2(kScreenSize));

    std::vector<float> sphericalHarmonics;
    int sphericalHarmonicsDegree{0};
    int sphericalHarmonicsCount{0};

#if false
    // Use randomly generated splats. Useful for testing and debugging.
    constexpr int32_t splatCount = 1 << 4;
    constexpr auto worldBoundsExtent = 4.0f;

    const auto boundsMin = glm::vec4(-worldBoundsExtent, -worldBoundsExtent, -worldBoundsExtent, 1.0f);
    const auto boundsMax = glm::vec4(worldBoundsExtent, worldBoundsExtent, worldBoundsExtent, 1.0f);

    auto positions = std::vector<float4>(splatCount);
    auto scales = std::vector<float4>(splatCount);
    auto rotations = std::vector<float4>(splatCount);
    auto colors = std::vector<float4>(splatCount);
    GenerateRandomGaussians(positions, scales, rotations, colors, 0.01f, 0.5f, boundsMin, boundsMax);

    camera_controls.SetBounds(glm::vec3(-worldBoundsExtent), glm::vec3(worldBoundsExtent));

#else
    // Load splats from a .ply file.
    std::vector<float4> positions;
    std::vector<float4> scales;
    std::vector<float4> rotations;
    std::vector<float4> colors;
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    auto splatCount = ParsePly(
        argv[1],
        positions,
        scales,
        rotations,
        colors,
        sphericalHarmonics,
        sphericalHarmonicsDegree,
        sphericalHarmonicsCount,
        boundsMin,
        boundsMax);
    auto hasExtraSphericalHarmonics = sphericalHarmonicsDegree > 1;

    std::vector<float> alignedSphericalHarmonics;
    if (sphericalHarmonicsDegree != 0)
    {
        RealignSphericalHarmonics(
            sphericalHarmonics, alignedSphericalHarmonics, 256, sphericalHarmonicsCount, splatCount);
        std::swap(sphericalHarmonics, alignedSphericalHarmonics);
    }

    camera_controls.SetBounds(boundsMin, boundsMax);

#endif

    // Create and compile the GLSL program. Used to draw the texture CUDA renders to.
    auto programTexturedId = CompileShadersProgram(kVertexShaderTextured, kFragmentShaderTextured);
    auto textureId = glGetUniformLocation(programTexturedId, "textureSampler");

    auto vertexArray = GLVertexArray();
    // OpenGL buffers.
    auto glVertexBuffer = GLBuffer<glm::vec3, GL_ARRAY_BUFFER>(4, kQuadVertices);
    auto glUvBuffer = GLBuffer<glm::vec2, GL_ARRAY_BUFFER>(4, kQuadUvs);
    auto glIndexBuffer = GLBuffer<uint32_t, GL_ARRAY_BUFFER>(6, kQuadIndices);
    auto glColorBuffer = GLBuffer<uchar4, GL_PIXEL_UNPACK_BUFFER>(kScreenSize * kScreenSize);

    // CUDA buffers.
    auto positionBuffer = DeviceBuffer<float4>(positions);
    auto scalesBuffer = DeviceBuffer<float4>(scales);
    auto rotationsBuffer = DeviceBuffer<float4>(rotations);
    auto colorBuffer = DeviceBuffer<float4>(colors);
    auto sphericalHarmonicsBuffer = DeviceBuffer<float>(sphericalHarmonics);
    auto conicAndColorBuffer = DeviceBuffer<float4>(splatCount);
    auto positionClipSpaceXYBuffer = DeviceBuffer<float2>(splatCount);
    auto positionClipSpaceZBuffer = DeviceBuffer<float>(splatCount);
    auto screenEllipseBuffer = DeviceBuffer<float4>(splatCount);
    auto tileRangeBuffer = DeviceBuffer<int32_t>(kTotalTiles * 2);

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
    auto colorBufferResource = CudaGraphicsResource(glColorBuffer.GetBufferId(), cudaGraphicsRegisterFlagsNone);

    Stats stats;
    long frameCount{0};
    auto cuda_timer = CudaTimer();

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
            tileListKeysCurrentBuffer.ResizeIfNeeded(tileListCapacity);
            tileListValuesCurrentBuffer.ResizeIfNeeded(tileListCapacity);
            tileListKeysAlternateBuffer.ResizeIfNeeded(tileListCapacity);
            tileListValuesAlternateBuffer.ResizeIfNeeded(tileListCapacity);
            tileListIsSaturated = false;
        }

        // Update time.
        auto time = glfwGetTime();
        deltaTime = time - lastTime;
        lastTime = time;

        // Update input & camera controls.
        camera_controls.Update((float) deltaTime);

        // Build camera data.
        CameraData cameraData;
        cameraData.position = camera_controls.GetPosition();
        cameraData.aspect = camera_controls.GetAspect();
        cameraData.projection = camera_controls.GetProjection();
        cameraData.viewProjection = camera_controls.GetViewProjection();
        cameraData.view = camera_controls.GetView();
        auto cotangentY = 1.0f / glm::tan(camera_controls.GetFieldOfView() * 0.5f);
        auto cotangentX = cotangentY / camera_controls.GetAspect();
        cameraData.fovCotangent = glm::vec2(cotangentX, cotangentY);
        // Orthographic mapping of the Z axis.
        // We use the default right handed coordinates, so we flip Z via scaleZ.
        // Clip depth is in [0, 1], this range matters when evaluating sorting keys.
        auto scaleZ = -2.0f / (camera_controls.GetFar() - camera_controls.GetNear());
        auto translationZ = -(camera_controls.GetFar() + camera_controls.GetNear())
                          / (camera_controls.GetFar() - camera_controls.GetNear());
        cameraData.depthScaleBias = glm::vec2(scaleZ, translationZ);

        // CUDA Update.
        {
            auto colorBinding = colorBufferResource.GetBinding<uchar4>();

            // Clear color buffer.
            CHECK_CUDA_ERRORS(cudaMemset(colorBinding.GetPtr(), 0, glColorBuffer.GetSizeInBytes()));

            // Clear tile ranges. Needed to identify empty ranges.
            // Set all pointers to 0xFFFFFFFF = -1
            tileRangeBuffer.ClearMemory(255);

            // Set global data.
            GlobalArgs globalArgs;
            globalArgs.primitiveCount = splatCount;
            globalArgs.positionsClipSpaceXY = positionClipSpaceXYBuffer.GetPtr();
            globalArgs.positionsClipSpaceZ = positionClipSpaceZBuffer.GetPtr();
            globalArgs.sphericalHarmonicsDegree = sphericalHarmonicsDegree;
            globalArgs.sphericalHarmonicsCount = sphericalHarmonicsCount;
            globalArgs.sphericalHarmonics = sphericalHarmonicsBuffer.GetPtr();
            globalArgs.screenEllipses = screenEllipseBuffer.GetPtr();
            globalArgs.positions = positionBuffer.GetPtr();
            globalArgs.scales = scalesBuffer.GetPtr();
            globalArgs.rotations = rotationsBuffer.GetPtr();
            globalArgs.colors = colorBuffer.GetPtr();
            globalArgs.conics = conicAndColorBuffer.GetPtr();
            globalArgs.tileRanges = tileRangeBuffer.GetPtr();
            globalArgs.backBuffer = colorBinding.GetPtr();
            globalArgs.cameraData = cameraData;

            SetGlobalArgs(&globalArgs);

            // Pass current tile list buffers.
            TileListArgs tileListArgs;
            tileListArgs.keys = tileListKeysCurrentBuffer.GetPtr();
            tileListArgs.values = tileListValuesCurrentBuffer.GetPtr();
            tileListArgs.capacity = tileListCapacity;

            SetTileListArgs(&tileListArgs);

            if (sphericalHarmonicsDegree != 0)
            {
                EvaluateSphericalHarmonics(cuda_timer, splatCount);
                stats.evaluate_spherical_harmonics += (double) cuda_timer.GetElapsedTimeMs();
            }

            EvaluateSplatClipData(cuda_timer, splatCount);
            stats.evaluate_clip_data += (double) cuda_timer.GetElapsedTimeMs();

            // Multiple blocks per multiprocessor, to give the scheduler a chance to hide memory latency.
            tileListArgs.size = BuildTileList(cuda_timer, prop.multiProcessorCount * 8, tileListCapacity);
            assert(tileListArgs.size <= tileListCapacity);
            stats.build_tile_list += (double) cuda_timer.GetElapsedTimeMs();

            // Track whether the tile list is saturated.
            tileListIsSaturated = tileListArgs.size == tileListCapacity;

            // Update render list args with render list size.
            SetTileListArgs(&tileListArgs);

            // Instantiate double buffers.
            DoubleBuffer<uint64_t> tileListKeys(
                tileListKeysCurrentBuffer.GetPtr(), tileListKeysAlternateBuffer.GetPtr());
            DoubleBuffer<int32_t> tileListValues(
                tileListValuesCurrentBuffer.GetPtr(), tileListValuesAlternateBuffer.GetPtr());

            SortTileList(
                cuda_timer, tileListArgs.size, deviceTempStorage, tempStorageSizeInBytes, tileListKeys, tileListValues);
            stats.sort_tile_list += (double) cuda_timer.GetElapsedTimeMs();

            // Update render list args with current buffers.
            tileListArgs.keys = tileListKeys.Current();
            tileListArgs.values = tileListValues.Current();

            SetTileListArgs(&tileListArgs);

            // Bypass if no splat is visible.
            if (tileListArgs.size != 0)
            {
                EvaluateTileRange(cuda_timer, tileListArgs.size);
                stats.evaluate_tile_ranges += (double) cuda_timer.GetElapsedTimeMs();
            }

            RasterizeTile(cuda_timer);
            stats.render_depth_buffer += (double) cuda_timer.GetElapsedTimeMs();

            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        // Draw the rendered texture.
        {
            glUseProgram(programTexturedId);
            glEnable(GL_TEXTURE_2D);

            auto vertexBinding = glVertexBuffer.GetBinding(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
            auto uvBinding = glUvBuffer.GetBinding(1);
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
            auto depthBinding = glColorBuffer.GetBinding();
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kScreenSize, kScreenSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

            glUniform1i(textureId, 0);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIndexBuffer.GetBufferId());
            glDrawElements(GL_TRIANGLES, glIndexBuffer.GetSize(), GL_UNSIGNED_INT, (void*) 0);

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
        CHECK_CUDA_ERRORS(cudaFree(deviceTempStorage));
    }

    glDeleteProgram(programTexturedId);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    // Evaluate profiling averages and print to terminal.
    stats.evaluate_spherical_harmonics /= (double) frameCount;
    stats.evaluate_clip_data /= (double) frameCount;
    stats.build_tile_list /= (double) frameCount;
    stats.sort_tile_list /= (double) frameCount;
    stats.evaluate_tile_ranges /= (double) frameCount;
    stats.render_depth_buffer /= (double) frameCount;

    auto totalMs = //
        stats.evaluate_clip_data + //
        stats.build_tile_list + //
        stats.sort_tile_list + //
        stats.evaluate_tile_ranges + //
        stats.render_depth_buffer; //

    printf("evaluateSphericalHarmonics average time ms: %2.6f\n", stats.evaluate_spherical_harmonics);
    printf("evaluateClipData average time ms: %2.6f\n", stats.evaluate_clip_data);
    printf("buildTileList average time ms: %2.6f\n", stats.build_tile_list);
    printf("sortTileList average time ms: %2.6f\n", stats.sort_tile_list);
    printf("evaluateTileRanges average time ms: %2.6f\n", stats.evaluate_tile_ranges);
    printf("renderDepthBuffer average time ms: %2.6f\n", stats.render_depth_buffer);
    printf("Total average time ms: %2.6f\n", totalMs);
    getchar();

    return 0;
}
