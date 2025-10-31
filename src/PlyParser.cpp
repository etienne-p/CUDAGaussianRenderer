#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include "PlyParser.h"

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// Match INRIA serialization.
struct PlyVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 sphericalHarmonics;
    float opacity;
    glm::vec3 scale;
    glm::vec4 rotation;
};

int ParsePly(const char* filePath, std::vector<PlyVertex>& vertices, glm::vec3& boundsMin, glm::vec3& boundsMax)
{
    std::ifstream infile(filePath, std::ios_base::binary);

    std::string lineStr;
    std::string placeholder;

    // Look for vertex count.
    while (std::getline(infile, lineStr))
    {
        if (lineStr.rfind("element", 0) == 0)
        {
            break;
        }
    }

    // Parse vertex count.
    std::stringstream ss(lineStr);
    int count;
    ss >> placeholder >> placeholder >> count;

    // We can allocate vertices.
    vertices.resize(count);

    // Look for header end.
    while (std::getline(infile, lineStr))
    {
        if (lineStr.compare("end_header") == 0)
        {
            break;
        }
    }

    // Deserialize all vertices.
    infile.read((char*) vertices.data(), count * sizeof(PlyVertex));

    boundsMin = glm::vec3(std::numeric_limits<float>::max());
    boundsMax = glm::vec3(std::numeric_limits<float>::min());

    for (auto& v : vertices)
    {
        boundsMin = glm::min(boundsMin, v.position);
        boundsMax = glm::max(boundsMax, v.position);
        v.rotation = glm::normalize(v.rotation);
        v.opacity = sigmoid(v.opacity);
        v.scale = glm::exp(v.scale);
    }

    return count;
}

int ParsePly(const char* filePath,
             std::vector<float4>& position,
             std::vector<float4>& covariance,
             std::vector<float4>& color,
             glm::vec3& boundsMin,
             glm::vec3& boundsMax)
{
    std::vector<PlyVertex> plyVertices;
    auto count = ParsePly(filePath, plyVertices, boundsMin, boundsMax);

    position.resize(count);
    covariance.resize(count * 2);
    color.resize(count);

    for (auto i = 0; i != count; ++i)
    {
        auto v = plyVertices[i];

        auto scale3x3 = glm::mat3(0);
        scale3x3[0][0] = v.scale[0];
        scale3x3[1][1] = v.scale[1];
        scale3x3[2][2] = v.scale[2];

        auto rot = glm::quat::wxyz(v.rotation.x, v.rotation.y, v.rotation.z, v.rotation.w);
        auto RS = glm::mat3_cast(rot) * scale3x3;
        auto cov = RS * glm::transpose(RS);

        // Covariance is symmetric.
        auto cov0 = float4{cov[0][0], cov[1][0], cov[2][0], 0};
        auto cov1 = float4{cov[1][1], cov[2][1], cov[2][2], 0};
        covariance[i * 2 + 0] = cov0;
        covariance[i * 2 + 1] = cov1;

        position[i] = float4{v.position.x, v.position.y, v.position.z, 0};

        // Convert spherical harmonics to RGB values.
        const float SH_C0 = 0.28209479177387814;
        auto rgb = v.sphericalHarmonics * SH_C0 + 0.5f;
        color[i] = float4{rgb.x, rgb.y, rgb.z, v.opacity};
    }

    return count;
}
