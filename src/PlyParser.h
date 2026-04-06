#pragma once

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector_types.h>

int parsePly(const char* filePath,
             std::vector<float4>& positions,
             std::vector<float4>& scales,
             std::vector<float4>& rotations,
             std::vector<float4>& colors,
             std::vector<float>& sphericalHarmonics,
             int& sphericalHarmonicsDegree,
             int& sphericalHarmonicsCount,
             glm::vec3& boundsMin,
             glm::vec3& boundsMax);
