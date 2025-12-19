#pragma once

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector_types.h>

uint32_t encodeVec4(const glm::vec4& v);

int parsePly(const char* filePath,
             std::vector<float4>& position,
             std::vector<float4>& scaleAndRotation,
             std::vector<float4>& color,
             std::vector<float>& sphericalHarmonics,
             int& sphericalHarmonicsDegree,
             int& sphericalHarmonicsCount,
             glm::vec3& boundsMin,
             glm::vec3& boundsMax);
