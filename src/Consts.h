#pragma once
#include <cstdint>

constexpr int32_t k_ScreenSize = 1024;
constexpr int32_t k_TileSize = 16;
constexpr int32_t k_FrustumCornersCount = 8;
constexpr int32_t k_TilesPerScreen = k_ScreenSize / k_TileSize;
constexpr int32_t k_TotalTiles = k_TilesPerScreen * k_TilesPerScreen;
