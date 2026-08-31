// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Triton Ascend C-interface wrappers for SHMEM 1.6's low-level PIPE_S UDMA.

#include <cstdint>

#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/shmem_device_cc.h"
#include "device/gm2gm/engine/shmem_device_udma.h"

template <typename T>
struct MemRef1D {
    __gm__ T* allocated;
    __gm__ T* aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
};

extern "C" __aicore__ void _mlir_ciface_aclshmemi_udma_put_nbi(
    MemRef1D<bfloat16_t>* destination,
    MemRef1D<bfloat16_t>* source,
    uint32_t num_elements,
    int32_t destination_rank)
{
    __gm__ bfloat16_t* destination_ptr =
        destination->aligned + destination->offset;
    __gm__ bfloat16_t* source_ptr = source->aligned + source->offset;
    aclshmemi_udma_put_nbi(
        destination_ptr,
        source_ptr,
        num_elements,
        destination_rank);
}

extern "C" __aicore__ void _mlir_ciface_aclshmemx_udma_put_signal_nbi(
    MemRef1D<bfloat16_t>* destination,
    MemRef1D<bfloat16_t>* source,
    uint32_t num_elements,
    MemRef1D<int32_t>* signal_address,
    int32_t signal,
    int32_t destination_rank)
{
    __gm__ bfloat16_t* destination_ptr =
        destination->aligned + destination->offset;
    __gm__ bfloat16_t* source_ptr = source->aligned + source->offset;
    __gm__ int32_t* signal_ptr =
        signal_address->aligned + signal_address->offset;
    // The UDMA WRITE_WITH_NOTIFY WQE updates one aligned uint64 signal word.
    // Replica slots reserve 16 int32 values and FC1 observes the low word, so
    // a positive int32 epoch is ABI-compatible with the 64-bit notify field.
    aclshmemx_udma_put_signal_nbi(
        destination_ptr,
        source_ptr,
        num_elements,
        reinterpret_cast<__gm__ uint64_t*>(signal_ptr),
        static_cast<uint64_t>(signal),
        destination_rank);
}

extern "C" __aicore__ void _mlir_ciface_aclshmemx_udma_quiet(
    int32_t destination_rank)
{
    aclshmemx_udma_quiet(destination_rank);
}
