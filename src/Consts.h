#pragma once

#include <cstdint>

constexpr int32_t kScreenSize = 1024;
constexpr int32_t kTileSize = 16;
constexpr int32_t kFrustumCornersCount = 8;
constexpr int32_t kTilesPerScreen = kScreenSize / kTileSize;
constexpr int32_t kTotalTiles = kTilesPerScreen * kTilesPerScreen;
