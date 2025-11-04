// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/cuda/hardware/cuda_memory_allocator.hpp>

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

#include "cuda_memory_block_allocator.hpp"

namespace xmipp4 
{
namespace hardware
{

class cuda_device_memory_resource;

class cuda_device_memory_allocator final
    : public cuda_memory_allocator
{
public:
    cuda_device_memory_allocator(cuda_device_memory_resource &resource);
    cuda_device_memory_allocator(
        const cuda_device_memory_allocator &other
    ) = delete;
    cuda_device_memory_allocator(
        cuda_device_memory_allocator &&other
    ) = default;
    ~cuda_device_memory_allocator() override = default;

    cuda_device_memory_allocator&
    operator=(const cuda_device_memory_allocator &other) = delete;
    cuda_device_memory_allocator&
    operator=(cuda_device_memory_allocator &&other) = default;

    cuda_memory_resource& get_memory_resource() const noexcept override;

    std::shared_ptr<cuda_buffer> allocate(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue
    ) override;

    std::shared_ptr<buffer> allocate(
        std::size_t size, 
        std::size_t alignment, 
        device_queue *queue
    ) override;

private:
    cuda_memory_block_allocator m_allocator;

};

} // namespace hardware
} // namespace xmipp4
