#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector_types.h>

int ParsePly(const char* filePath,
             std::vector<float4>& position,
             std::vector<float4>& covariance,
             std::vector<float4>& color,
             glm::vec3& boundsMin,
             glm::vec3& boundsMax);
