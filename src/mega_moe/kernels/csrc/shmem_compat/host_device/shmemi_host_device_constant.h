/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Compatibility copy omitted from the SHMEM 1.6 Python wheel.
 */
#ifndef SHMEMI_HOST_DEVICE_CONSTANT_H
#define SHMEMI_HOST_DEVICE_CONSTANT_H

#include <cstdint>

namespace shm {

constexpr uint32_t UDMA_CQ_DEPTH_DEFAULT = 32768;
constexpr uint32_t UDMA_SQ_DEPTH_DEFAULT = 8192;
constexpr uint32_t UDMA_RQ_DEPTH_DEFAULT = 256;
constexpr uint32_t UDMA_MAX_SQE_BB_NUM = 4;
constexpr uint32_t UDMA_SQ_BASKBLK_CNT =
    UDMA_SQ_DEPTH_DEFAULT * UDMA_MAX_SQE_BB_NUM;

} // namespace shm

#endif
