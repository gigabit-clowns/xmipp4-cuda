// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_device_buffer.hpp"

#include "cuda_device_memory_resource.hpp"

namespace xmipp4
{
namespace hardware
{

cuda_device_buffer::cuda_device_buffer(
    std::size_t size, 
    std::size_t alignment, 
    cuda_device_queue *queue, 
    cuda_memory_block_allocator &allocator
)
    : m_allocation(size, alignment, queue, allocator)
{
}

void* cuda_device_buffer::get_device_ptr() noexcept
{
    const auto *block = m_allocation.get_memory_block();
    if (block)
    {
        return block->get_data_ptr();
    }
    else
    {
        return nullptr;
    }
}

const void* cuda_device_buffer::get_device_ptr() const noexcept
{
    const auto *block = m_allocation.get_memory_block();
    if (block)
    {
        return block->get_data_ptr();
    }
    else
    {
        return nullptr;
    }
}

void* cuda_device_buffer::get_host_ptr() noexcept
{
    return nullptr; // Not device accessible.
}

const void* cuda_device_buffer::get_host_ptr() const noexcept
{
    return nullptr; // Not device accessible.
}

std::size_t cuda_device_buffer::get_size() const noexcept
{
    const auto *block = m_allocation.get_memory_block();
    if (block)
    {
        return block->get_size();
    }
    else
    {
        return 0UL;
    }
}

cuda_memory_resource& 
cuda_device_buffer::get_memory_resource() const noexcept
{
    return m_allocation.get_allocator().get_memory_resource();
}

void cuda_device_buffer::record_queue(device_queue &queue, bool)
{
    m_allocation.record_queue(dynamic_cast<cuda_device_queue&>(queue));
}

} // namespace hardware
} // namespace xmipp4
