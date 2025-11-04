// SPDX-License-Identifier: GPL-3.0-only

#pragma once
#include "cuda_memory_block.hpp"
#include "cuda_memory_block_pool.hpp"

#include <xmipp4/core/span.hpp>

#include <xmipp4/cuda/hardware/cuda_event.hpp>
#include <xmipp4/cuda/hardware/cuda_device_queue.hpp>

#include <forward_list>
#include <utility>
#include <vector>

namespace xmipp4 
{
namespace hardware
{

/**
 * @brief Handles deferred deallocations to allow mixing multiple
 * CUDA queues (streams).
 * 
 */
class cuda_memory_block_deferred_release
{
public:
    cuda_memory_block_deferred_release() = default;
    cuda_memory_block_deferred_release(
        const cuda_memory_block_deferred_release &other
    ) = delete;
    cuda_memory_block_deferred_release(
        cuda_memory_block_deferred_release &&other
    ) = default;
    ~cuda_memory_block_deferred_release() = default;

    cuda_memory_block_deferred_release&
    operator=(const cuda_memory_block_deferred_release &other) = delete;
    cuda_memory_block_deferred_release&
    operator=(cuda_memory_block_deferred_release &&other) = default;

    /**
     * @brief Iterate through release events and return back all blocks
     * that have no pending events.
     * 
     * @param cache The cache from which all blocks were allocated.
     * 
     */
    void process_pending_free(cuda_memory_block_pool &cache);

    /**
     * @brief Signal events for each of the provided CUDA queues for a 
     * given block.
     * 
     * @param ite Iterator to the block. Must be dereferenceable.
     * @param other_queues Queues that need to be processed for actually
     * freeing the block.
     * 
     * @note This function does not check wether ite has been provided 
     * previously. Calling it twice with the same block before it has
     * been returned to the pool leads to undefined behavior.
     * 
     */
    void signal_events(
        cuda_memory_block_pool::iterator ite,
        span<cuda_device_queue *const> other_queues 
    );

private:
    using event_list = std::forward_list<cuda_event>;
    using block_key = cuda_memory_block_pool::iterator;

    event_list m_event_pool;
    std::vector<std::pair<block_key, event_list>> m_pending_free;

    /**
     * @brief Pop all signaled events from the list.
     * 
     * @param events Event list from which completed events are popt.
     */
    void pop_completed_events(event_list &events);

}; 

} // namespace hardware
} // namespace xmipp4
