#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <array>
#include <stdexcept>
#include "PlyParser.h"

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

void parseHeader(std::ifstream& inFile, std::vector<std::string>& properties, int& vertexCount)
{
    vertexCount = -1;

    std::string lineStr;
    std::string placeholder;

    std::string line;
    auto littleEndian{false};

    // Prevent infinite loop with bad headers.
    constexpr int maxLength{256};
    auto iteration{0};

    // Iterate over lines.
    while (std::getline(inFile, line))
    {
        std::stringstream ss(line);
        std::string word;

        // Iterate over words within lines.
        // the normal input operator >> separates on whitespace and so can be used to read "words"
        if (!(ss >> word))
        {
            throw std::invalid_argument("Unexpected empty line.");
        }

        // TODO: Account for the expected order of appearance.

        if (word == "ply")
        {
            // We expect this but have nothing to do with it.
            continue;
        }

        else if (word == "format")
        {
            littleEndian = ss >> word && word == "binary_little_endian";
        }

        else if (word == "element")
        {
            // We expect 2 words, the vertex type and its count.
            if (!(ss >> word && word == "vertex"))
            {
                throw std::invalid_argument("Unexpected element type.");
            }

            if (!(ss >> word))
            {
                throw std::invalid_argument("Unexpected element count.");
            }

            vertexCount = std::stoi(word);
        }

        else if (word == "property")
        {
            // We only expect float for the time being.
            if (!(ss >> word && word == "float"))
            {
                throw std::invalid_argument("Unexpected property format, expected float.");
            }

            if (!(ss >> word))
            {
                throw std::invalid_argument("Unexpected property, missing name.");
            }

            if (std::find(properties.begin(), properties.end(), word) != properties.end())
            {
                throw std::invalid_argument("Duplicated property \"" + word + "\".");
            }

            properties.push_back(word);
        }

        else if (word == "end_header")
        {
            if (!littleEndian || vertexCount == -1)
            {
                throw std::invalid_argument("Invalid header. Format or vertex count not found.");
            }

            break;
        }

        ++iteration;

        if (iteration > maxLength)
        {
            throw std::invalid_argument("Invalid header. End not found.");
        }
    }
}

int parsePly(const char* filePath, std::vector<std::string>& properties, std::vector<std::vector<float>>& propertiesData)
{
    assert(properties.size() == 0);
    assert(propertiesData.size() == 0);

    std::ifstream inFile(filePath, std::ios_base::binary);

    // Parse header.
    auto vertexCount{-1};
    parseHeader(inFile, properties, vertexCount);

    const auto numProps = properties.size();
    // Allocate memory;
    propertiesData.resize(numProps);
    for (auto i = 0; i != numProps; ++i)
    {
        propertiesData[i] = std::vector<float>();
        propertiesData[i].resize(vertexCount);
    }

    // Deserialize properties data.
    for (auto i = 0; i != vertexCount; ++i)
    {
        for (auto j = 0; j != numProps; ++j)
        {
            float val;
            inFile.read(reinterpret_cast<char*>(&val), sizeof(float));
            propertiesData[j][i] = val;
        }
    }

    return vertexCount;
}

int indexOf(const std::vector<std::string>& properties, const std::string& property)
{
    auto it = std::find(properties.begin(), properties.end(), property);
    if (it == properties.end())
    {
        throw std::invalid_argument("Missing property \"" + property + "\".");
    }
    return std::distance(properties.begin(), it);
}

uint32_t encodeVec4(const glm::vec4& v)
{
    auto clamped = glm::clamp(v);
    // clang-format off
    return (
        ((uint32_t) (clamped.x * 255.0f) << 24u) | 
        ((uint32_t) (clamped.y * 255.0f) << 16u) | 
        ((uint32_t) (clamped.z * 255.0f) << 8u ) | 
        ((uint32_t) (clamped.w * 255.0f)       ));
    // clang-format on
}

int parsePly(const char* filePath,
             std::vector<float4>& position,
             std::vector<float4>& scaleAndRotation,
             std::vector<float4>& color,
             std::vector<float>& sphericalHarmonics,
             int& sphericalHarmonicsDegree,
             int& sphericalHarmonicsCount,
             glm::vec3& boundsMin,
             glm::vec3& boundsMax)
{
    std::vector<std::string> properties;
    std::vector<std::vector<float>> propertiesData;
    auto vertexCount = parsePly(filePath, properties, propertiesData);

    // Ensure that expected (required) properties are present and store their offset.
    std::array<int, 14> offsets{};
    offsets[0] = indexOf(properties, "x");
    offsets[1] = indexOf(properties, "y");
    offsets[2] = indexOf(properties, "z");
    offsets[3] = indexOf(properties, "rot_0");
    offsets[4] = indexOf(properties, "rot_1");
    offsets[5] = indexOf(properties, "rot_2");
    offsets[6] = indexOf(properties, "rot_3");
    offsets[7] = indexOf(properties, "scale_0");
    offsets[8] = indexOf(properties, "scale_1");
    offsets[9] = indexOf(properties, "scale_2");
    offsets[10] = indexOf(properties, "f_dc_0");
    offsets[11] = indexOf(properties, "f_dc_1");
    offsets[12] = indexOf(properties, "f_dc_2");
    offsets[13] = indexOf(properties, "opacity");

    // Allocate result memory.
    position.resize(vertexCount);
    scaleAndRotation.resize(vertexCount);
    color.resize(vertexCount);

    // Handle extra spherical harmonics.
    auto extraSphericalHarmonicsCount{0};
    auto shOffsets = std::vector<int>();
    for (;;)
    {
        auto it =
            std::find(properties.begin(), properties.end(), "f_rest_" + std::to_string(extraSphericalHarmonicsCount));
        if (it == properties.end())
        {
            break;
        }
        else
        {
            shOffsets.push_back(std::distance(properties.begin(), it));
        }

        ++extraSphericalHarmonicsCount;
    }

    // Count spherical harmonics levels.
    auto expectedSphericalHarmonicsCount{0};
    sphericalHarmonicsDegree = 0;
    while (expectedSphericalHarmonicsCount < extraSphericalHarmonicsCount)
    {
        // There are (2l + 1) spherical harmonics for each degree (level).
        // Add one since level zero is the color stored in f_dc_{0, 2}.
        // Times 3 as we have RGB channels.
        expectedSphericalHarmonicsCount += (2 * (sphericalHarmonicsDegree + 1) + 1) * 3;
        ++sphericalHarmonicsDegree;
    }

    // Verify that the number of spherical harmonics is sensible.
    if (expectedSphericalHarmonicsCount != extraSphericalHarmonicsCount)
    {
        throw std::invalid_argument(
            "Expected degree " + std::to_string(sphericalHarmonicsDegree) + ", "
            + std::to_string(expectedSphericalHarmonicsCount) + " extra spherical harmonics." + " Found "
            + std::to_string(extraSphericalHarmonicsCount) + ".");
    }

    sphericalHarmonicsCount = extraSphericalHarmonicsCount + 3;

    if (sphericalHarmonicsDegree != 0)
    {
        // We align spherical harmonics using 4 floats for aligned access.
        sphericalHarmonics.resize(vertexCount * sphericalHarmonicsCount);

        for (auto i = 0; i != vertexCount; ++i)
        {
            auto ptr = reinterpret_cast<float*>(&(*sphericalHarmonics.begin()) + sphericalHarmonicsCount * i);

            // When using extra spherical harmonics we also bundle level 0 into the buffer.
            for (auto j = 0; j != 3; ++j)
            {
                *ptr = propertiesData[offsets[10 + j]][i];
                ++ptr;
            }

            // The rest of extra spherical harmonics, the ones that are optional.
            for (auto j = 0; j != extraSphericalHarmonicsCount; ++j)
            {
                *ptr = propertiesData[shOffsets[j]][i];
                ++ptr;
            }
        }

        auto rgbShCount = extraSphericalHarmonicsCount / 3;
        auto tmp = std::vector<float>(extraSphericalHarmonicsCount);

        // TODO: Merge re-order with step above (copy).
        for (auto i = 0; i != vertexCount; ++i)
        {
            auto start = sphericalHarmonicsCount * i + 3;
            for (auto j = 0; j != rgbShCount; ++j)
            {
                tmp[j * 3 + 0] = sphericalHarmonics[start + j];
                tmp[j * 3 + 1] = sphericalHarmonics[start + rgbShCount + j];
                tmp[j * 3 + 2] = sphericalHarmonics[start + rgbShCount * 2 + j];
            }
            for (auto j = 0; j != extraSphericalHarmonicsCount; ++j)
            {
                sphericalHarmonics[start + j] = tmp[j];
            }
        }
    }

    boundsMin = glm::vec3(std::numeric_limits<float>::max());
    boundsMax = glm::vec3(std::numeric_limits<float>::min());

    for (auto i = 0; i != vertexCount; ++i)
    {
        // Note: quaternion is encoded with .w first.
        // clang-format off
        auto translation = glm::vec3(
            propertiesData[offsets[0]][i], 
            propertiesData[offsets[1]][i], 
            propertiesData[offsets[2]][i]);
        auto rotation = glm::quat::wxyz(
            propertiesData[offsets[3]][i],
            propertiesData[offsets[4]][i],
            propertiesData[offsets[5]][i],
            propertiesData[offsets[6]][i]);
        auto scale = glm::vec3(
            propertiesData[offsets[7]][i], 
            propertiesData[offsets[8]][i], 
            propertiesData[offsets[9]][i]);
        auto sphericalHarmonics = glm::vec3(
            propertiesData[offsets[10]][i], 
            propertiesData[offsets[11]][i], 
            propertiesData[offsets[12]][i]);
        auto opacity = propertiesData[offsets[13]][i];
        // clang-format on

        // Process deserialized data.
        rotation = glm::normalize(rotation);
        scale = glm::exp(scale);
        opacity = sigmoid(opacity);

        // Update bounds.
        boundsMin = glm::min(boundsMin, translation);
        boundsMax = glm::max(boundsMax, translation);

        // Convert spherical harmonics to RGB values.
        const float SH_C0{0.28209479177387814f};
        auto rgb = sphericalHarmonics * SH_C0 + 0.5f;

        // Quantize rotation.
        auto quantizedRotation = encodeVec4((glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w) + 1.0f) * 0.5f);

        // Bundle opacity with position. Comes in handy when evaluating spherical harmonics.
        position[i] = float4{translation.x, translation.y, translation.z, opacity};
        scaleAndRotation[i] = float4{scale.x, scale.y, scale.z, reinterpret_cast<float&>(quantizedRotation)};
        color[i] = float4{rgb.x, rgb.y, rgb.z, opacity};
    }

    return vertexCount;
}
