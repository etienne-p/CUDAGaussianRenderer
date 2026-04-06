#include "PlyParser.h"

#include <cassert>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

static float Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static void ParseHeader(std::ifstream& in_file, std::vector<std::string>& properties, int& vertex_count)
{
    vertex_count = -1;

    std::string line;
    auto little_endian{false};

    constexpr int kMaxHeaderLines{256};
    auto iteration{0};

    while (std::getline(in_file, line))
    {
        std::stringstream ss(line);
        std::string word;

        if (!(ss >> word))
        {
            throw std::invalid_argument("Unexpected empty line.");
        }

        if (word == "ply")
        {
            continue;
        }
        else if (word == "format")
        {
            little_endian = ss >> word && word == "binary_little_endian";
        }
        else if (word == "element")
        {
            if (!(ss >> word && word == "vertex"))
            {
                throw std::invalid_argument("Unexpected element type.");
            }

            if (!(ss >> word))
            {
                throw std::invalid_argument("Unexpected element count.");
            }

            vertex_count = std::stoi(word);
        }
        else if (word == "property")
        {
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
            if (!little_endian || vertex_count == -1)
            {
                throw std::invalid_argument("Invalid header. Format or vertex count not found.");
            }

            break;
        }

        ++iteration;

        if (iteration > kMaxHeaderLines)
        {
            throw std::invalid_argument("Invalid header. End not found.");
        }
    }
}

static int ParsePlyRaw(
    const char* file_path, std::vector<std::string>& properties, std::vector<std::vector<float>>& properties_data)
{
    assert(properties.size() == 0);
    assert(properties_data.size() == 0);

    std::ifstream in_file(file_path, std::ios_base::binary);

    auto vertex_count{-1};
    ParseHeader(in_file, properties, vertex_count);

    const auto num_props = properties.size();
    properties_data.resize(num_props);
    for (auto i = 0; i != num_props; ++i)
    {
        properties_data[i] = std::vector<float>();
        properties_data[i].resize(vertex_count);
    }

    for (auto i = 0; i != vertex_count; ++i)
    {
        for (auto j = 0; j != num_props; ++j)
        {
            float val;
            in_file.read(reinterpret_cast<char*>(&val), sizeof(float));
            properties_data[j][i] = val;
        }
    }

    return vertex_count;
}

static int IndexOf(const std::vector<std::string>& properties, const std::string& property)
{
    auto it = std::find(properties.begin(), properties.end(), property);
    if (it == properties.end())
    {
        throw std::invalid_argument("Missing property \"" + property + "\".");
    }
    return static_cast<int>(std::distance(properties.begin(), it));
}

int ParsePly(const char* file_path,
             std::vector<float4>& positions,
             std::vector<float4>& scales,
             std::vector<float4>& rotations,
             std::vector<float4>& colors,
             std::vector<float>& spherical_harmonics,
             int& spherical_harmonics_degree,
             int& spherical_harmonics_count,
             glm::vec3& bounds_min,
             glm::vec3& bounds_max)
{
    std::vector<std::string> properties;
    std::vector<std::vector<float>> properties_data;
    auto vertex_count = ParsePlyRaw(file_path, properties, properties_data);

    std::array<int, 14> offsets{};
    offsets[0] = IndexOf(properties, "x");
    offsets[1] = IndexOf(properties, "y");
    offsets[2] = IndexOf(properties, "z");
    offsets[3] = IndexOf(properties, "rot_0");
    offsets[4] = IndexOf(properties, "rot_1");
    offsets[5] = IndexOf(properties, "rot_2");
    offsets[6] = IndexOf(properties, "rot_3");
    offsets[7] = IndexOf(properties, "scale_0");
    offsets[8] = IndexOf(properties, "scale_1");
    offsets[9] = IndexOf(properties, "scale_2");
    offsets[10] = IndexOf(properties, "f_dc_0");
    offsets[11] = IndexOf(properties, "f_dc_1");
    offsets[12] = IndexOf(properties, "f_dc_2");
    offsets[13] = IndexOf(properties, "opacity");

    positions.resize(vertex_count);
    scales.resize(vertex_count);
    rotations.resize(vertex_count);
    colors.resize(vertex_count);

    auto extra_spherical_harmonics_count{0};
    auto sh_offsets = std::vector<int>();
    for (;;)
    {
        auto it = std::find(
            properties.begin(), properties.end(), "f_rest_" + std::to_string(extra_spherical_harmonics_count));
        if (it == properties.end())
        {
            break;
        }
        sh_offsets.push_back(static_cast<int>(std::distance(properties.begin(), it)));
        ++extra_spherical_harmonics_count;
    }

    auto expected_spherical_harmonics_count{0};
    spherical_harmonics_degree = 0;
    while (expected_spherical_harmonics_count < extra_spherical_harmonics_count)
    {
        expected_spherical_harmonics_count += (2 * (spherical_harmonics_degree + 1) + 1) * 3;
        ++spherical_harmonics_degree;
    }

    if (expected_spherical_harmonics_count != extra_spherical_harmonics_count)
    {
        throw std::invalid_argument(
            "Expected degree " + std::to_string(spherical_harmonics_degree) + ", "
            + std::to_string(expected_spherical_harmonics_count) + " extra spherical harmonics." + " Found "
            + std::to_string(extra_spherical_harmonics_count) + ".");
    }

    spherical_harmonics_count = extra_spherical_harmonics_count + 3;

    if (spherical_harmonics_degree != 0)
    {
        spherical_harmonics.resize(vertex_count * spherical_harmonics_count);

        for (auto i = 0; i != vertex_count; ++i)
        {
            auto ptr = reinterpret_cast<float*>(&(*spherical_harmonics.begin()) + spherical_harmonics_count * i);

            for (auto j = 0; j != 3; ++j)
            {
                *ptr = properties_data[offsets[10 + j]][i];
                ++ptr;
            }

            for (auto j = 0; j != extra_spherical_harmonics_count; ++j)
            {
                *ptr = properties_data[sh_offsets[j]][i];
                ++ptr;
            }
        }

        auto rgb_sh_count = extra_spherical_harmonics_count / 3;
        auto tmp = std::vector<float>(extra_spherical_harmonics_count);

        for (auto i = 0; i != vertex_count; ++i)
        {
            auto start = spherical_harmonics_count * i + 3;
            for (auto j = 0; j != rgb_sh_count; ++j)
            {
                tmp[j * 3 + 0] = spherical_harmonics[start + j];
                tmp[j * 3 + 1] = spherical_harmonics[start + rgb_sh_count + j];
                tmp[j * 3 + 2] = spherical_harmonics[start + rgb_sh_count * 2 + j];
            }
            for (auto j = 0; j != extra_spherical_harmonics_count; ++j)
            {
                spherical_harmonics[start + j] = tmp[j];
            }
        }
    }

    bounds_min = glm::vec3(std::numeric_limits<float>::max());
    bounds_max = glm::vec3(std::numeric_limits<float>::lowest());

    constexpr float kShC0 = 0.28209479177387814f;

    for (auto i = 0; i != vertex_count; ++i)
    {
        // Note: quaternion is encoded with .w first.
        auto translation =
            glm::vec3(properties_data[offsets[0]][i], properties_data[offsets[1]][i], properties_data[offsets[2]][i]);
        auto rotation = glm::quat::wxyz(
            properties_data[offsets[3]][i],
            properties_data[offsets[4]][i],
            properties_data[offsets[5]][i],
            properties_data[offsets[6]][i]);
        auto scale =
            glm::vec3(properties_data[offsets[7]][i], properties_data[offsets[8]][i], properties_data[offsets[9]][i]);
        auto sh_rgb = glm::vec3(
            properties_data[offsets[10]][i], properties_data[offsets[11]][i], properties_data[offsets[12]][i]);
        auto opacity = properties_data[offsets[13]][i];

        rotation = glm::normalize(rotation);
        scale = glm::exp(scale);
        opacity = Sigmoid(opacity);

        bounds_min = glm::min(bounds_min, translation);
        bounds_max = glm::max(bounds_max, translation);

        auto rgb = sh_rgb * kShC0 + 0.5f;

        positions[i] = float4{translation.x, translation.y, translation.z, opacity};
        scales[i] = float4{scale.x, scale.y, scale.z, 0.f};
        rotations[i] = float4{rotation.x, rotation.y, rotation.z, rotation.w};
        colors[i] = float4{rgb.x, rgb.y, rgb.z, opacity};
    }

    return vertex_count;
}
