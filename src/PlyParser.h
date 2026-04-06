#pragma once

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector_types.h>

int ParsePly(const char* file_path,
             std::vector<float4>& positions,
             std::vector<float4>& scales,
             std::vector<float4>& rotations,
             std::vector<float4>& colors,
             std::vector<float>& spherical_harmonics,
             int& spherical_harmonics_degree,
             int& spherical_harmonics_count,
             glm::vec3& bounds_min,
             glm::vec3& bounds_max);
