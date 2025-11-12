// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_transfer_host_to_device.hpp"

#include <xmipp4/cuda/hardware/cuda_device_queue.hpp>
#include <xmipp4/cuda/hardware/cuda_buffer.hpp>
#include <xmipp4/core/hardware/buffer.hpp>

#include "cuda_copy_bytes.hpp"

#include <stdexcept>

namespace xmipp4
{
namespace hardware
{

void cuda_memory_transfer_host_to_device::copy(
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
        source, 
        dynamic_cast<cuda_buffer&>(destination),
        regions,
        cuda_queue
    );
}

void cuda_memory_transfer_host_to_device::copy(
    const buffer &source, 
    cuda_buffer &destination,
    span<const copy_region> regions, 
    cuda_device_queue *queue
) const
{
    const auto *src_ptr = source.get_host_ptr();
    if (!src_ptr)
    {
        throw std::invalid_argument(
            "Source buffer is not host accessible."
        );
    }

    auto *dst_ptr = destination.get_device_ptr();
    if (!dst_ptr)
    {
        throw std::invalid_argument(
            "Destination buffer is not device accessible."
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
            cudaMemcpyHostToDevice,
            stream_handle
        );
    }
}
    
} // namespace hardware
} // namespace xmipp4
