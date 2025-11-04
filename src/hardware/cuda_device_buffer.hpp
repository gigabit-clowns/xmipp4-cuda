// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/cuda/hardware/cuda_buffer.hpp>

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

#include "cuda_memory_block_allocation.hpp"

namespace xmipp4 
{
namespace hardware
{

class cuda_device_buffer final
    : public cuda_buffer
{
public:
    cuda_device_buffer(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue, 
        cuda_memory_block_allocator &allocator
    );
    cuda_device_buffer(const cuda_device_buffer &other) = delete;
    cuda_device_buffer(cuda_device_buffer &&other) = default;
    ~cuda_device_buffer() override = default;

    cuda_device_buffer&
    operator=(const cuda_device_buffer &other) = delete;
    cuda_device_buffer&
    operator=(cuda_device_buffer &&other) = default;


    void* get_device_ptr() noexcept override;

    const void* get_device_ptr() const noexcept override;

    void* get_host_ptr() noexcept override;

    const void* get_host_ptr() const noexcept override;

    std::size_t get_size() const noexcept override;

    cuda_memory_resource& get_memory_resource() const noexcept override;

    void record_queue(device_queue &queue, bool exclusive=false) override;

private:
    cuda_memory_block_allocation m_allocation;

};

} // namespace hardware
} // namespace xmipp4
