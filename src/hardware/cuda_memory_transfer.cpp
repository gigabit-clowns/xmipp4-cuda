// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_transfer.hpp"

#include <xmipp4/cuda/hardware/cuda_device_queue.hpp>

#include <xmipp4/core/hardware/buffer.hpp>
#include <xmipp4/core/hardware/copy_region.hpp>
#include <xmipp4/core/hardware/device_queue.hpp>
#include <xmipp4/core/memory/byte.hpp>

#include <stdexcept>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace hardware
{

static
void copy_bytes(
    const memory::byte *src_ptr, 
    std::size_t src_size, 
    memory::byte* dst_ptr, 
    std::size_t dst_size,
    const copy_region &region,
    cudaMemcpyKind direction,
    cudaStream_t stream
)
{
    const auto src_offset = region.get_source_offset();
    const auto dst_offset = region.get_destination_offset();
    const auto byte_count = region.get_count();

    if (src_offset + byte_count > src_size)
    {
        throw std::out_of_range(
            "Copy region exceeds source buffer size."
        );
    }

    if (dst_offset + byte_count > dst_size)
    {
        throw std::out_of_range(
            "Copy region exceeds destination buffer size."
        );
    }

    cudaMemcpyAsync(
        dst_ptr+dst_offset, 
        src_ptr+src_offset, 
        byte_count,
        direction,
        stream
    );
}

void cuda_memory_transfer::copy(
    const buffer &source, 
    buffer &destination,
    span<const copy_region> regions, 
    cuda_device_queue *queue
) const
{
    const auto *src_ptr = static_cast<const memory::byte*>(source.get_host_ptr()); // TODO
    if (!src_ptr)
    {
        throw std::invalid_argument(
            "Source buffer is not host accessible."
        );
    }

    auto *dst_ptr = static_cast<memory::byte*>(destination.get_host_ptr()); // TODO
    if (!dst_ptr)
    {
        throw std::invalid_argument(
            "Destination buffer is not host accessible."
        );
    }

    cudaStream_t stream_handle = 0;
    if (queue)
    {
        stream_handle = queue->get_handle();
    }

    const auto src_size = source.get_size();
    const auto dst_size = destination.get_size();
    for (const auto &region : regions)
    {
        copy_bytes(
            src_ptr, src_size, 
            dst_ptr, dst_size, 
            region, 
            cudaMemcpyHostToDevice, //TODO m_direction
            stream_handle
        );
    }
}
    
} // namespace hardware
} // namespace xmipp4
