// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_device_to_host_memory_transfer.hpp"

#include <xmipp4/cuda/hardware/cuda_device_queue.hpp>
#include <xmipp4/cuda/hardware/cuda_buffer.hpp>
#include <xmipp4/core/hardware/buffer.hpp>

#include "cuda_copy_bytes.hpp"

#include <stdexcept>

namespace xmipp4
{
namespace hardware
{

void cuda_device_to_host_memory_transfer::copy(
    const buffer &source, 
    buffer &destination,
    span<const copy_region> regions, 
    device_queue *queue
) const
{
    cuda_device_queue *cuda_queue = nullptr;
    if (queue)
    {
        cuda_queue = &dynamic_cast<cuda_device_queue&>(*queue);
    }

    copy(
        dynamic_cast<const cuda_buffer&>(source),
        destination,
        regions,
        cuda_queue
    );
}

void cuda_device_to_host_memory_transfer::copy(
    const cuda_buffer &source, 
    buffer &destination,
    span<const copy_region> regions, 
    cuda_device_queue *queue
) const
{
    const auto *src_ptr = source.get_device_ptr();
    if (!src_ptr)
    {
        throw std::invalid_argument(
            "Source buffer is not device accessible."
        );
    }

    auto *dst_ptr = destination.get_host_ptr();
    if (!dst_ptr)
    {
        throw std::invalid_argument(
            "Destination buffer is not host accessible."
        );
    }

    cudaStream_t stream_handle = nullptr;
    if (queue)
    {
        stream_handle = queue->get_handle();
    }

    const auto src_size = source.get_size();
    const auto dst_size = destination.get_size();
    for (const auto &region : regions)
    {
        cuda_copy_bytes(
            src_ptr,
            src_size, 
            dst_ptr,
            dst_size, 
            region, 
            cudaMemcpyDeviceToHost,
            stream_handle
        );
    }
}
    
} // namespace hardware
} // namespace xmipp4
