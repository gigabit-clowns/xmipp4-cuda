// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <cuda_runtime.h>

namespace xmipp4 
{
namespace hardware
{

void cuda_copy_bytes(
    const void *src_ptr, 
    std::size_t src_size, 
    void* dst_ptr, 
    std::size_t dst_size,
    const copy_region &region,
    cudaMemcpyKind direction,
    cudaStream_t stream
);

} // namespace hardware
} // namespace xmipp4

#include "cuda_copy_bytes.inl"
