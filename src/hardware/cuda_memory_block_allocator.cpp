// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_block_allocator.hpp"

#include <stdexcept>

namespace xmipp4
{
namespace hardware
{

cuda_memory_block_allocator::cuda_memory_block_allocator(
    cuda_memory_resource &resource,
    std::size_t minimum_size, 
    std::size_t request_size_step 
)
    : m_resource(resource)
    , m_minimum_size(minimum_size)
    , m_request_size_step(request_size_step)
{
}

cuda_memory_resource& 
cuda_memory_block_allocator::get_memory_resource() const noexcept
{
    return m_resource;
}

void cuda_memory_block_allocator::release()
{
    m_deferred_blocks.process_pending_free(m_block_pool);
    release_blocks(m_block_pool, m_resource);
}

const cuda_memory_block& 
cuda_memory_block_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    const cuda_device_queue *queue,
    cuda_memory_block_usage_tracker **usage_tracker
) 
{
    auto *result = try_allocate(size, alignment, queue, usage_tracker);

    if (!result)
    {
        // Retry after releasing blocks
        release();
        result = try_allocate(size, alignment, queue, usage_tracker);
    }

    if (!result)
    {
        throw std::bad_alloc();
    }

    return *result;
}

const cuda_memory_block* 
cuda_memory_block_allocator::try_allocate(
    std::size_t size, 
    std::size_t alignment, 
    const cuda_device_queue *queue,
    cuda_memory_block_usage_tracker **usage_tracker
) 
{
    const cuda_memory_block *result;

    size = memory::align_ceil(size, m_request_size_step);
    m_deferred_blocks.process_pending_free(m_block_pool);
    const auto ite = allocate_block(
        m_block_pool,
        m_resource, 
        size,
        alignment,
        queue,
        m_minimum_size,
        m_request_size_step
    );

    if (ite != m_block_pool.end())
    {
        result = &(ite->first);
        if (usage_tracker)
        {
            *usage_tracker = &(ite->second.get_usage_tracker());
        }
    }
    else
    {
        result = nullptr;
        if (usage_tracker)
        {
            *usage_tracker = nullptr;
        }
    }
    
    return result;
}

void cuda_memory_block_allocator::deallocate(const cuda_memory_block &block)
{
    const auto ite = m_block_pool.find(block);
    if (ite == m_block_pool.end())
    {
        throw std::invalid_argument(
            "Provided block does not belong to the pool"
        );
    }

    const auto extra_queues = ite->second.get_usage_tracker().get_queues();
    if (extra_queues.empty())
    {
        deallocate_block(m_block_pool, ite);
    }
    else
    {
        m_deferred_blocks.signal_events(ite, extra_queues);
    }
}

} // namespace hardware
} // namespace xmipp4
