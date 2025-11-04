// SPDX-License-Identifier: GPL-3.0-only

#pragma once
#include "cuda_memory_block.hpp"
#include "cuda_memory_block_pool.hpp"
#include "cuda_memory_block_deferred_release.hpp"

namespace xmipp4 
{
namespace hardware
{

class cuda_memory_resource;

/**
 * @brief Manages a set of cuda_memory_block-s to efficiently
 * re-use them when possible.
 * 
 */
class cuda_memory_block_allocator
{
public:
    cuda_memory_block_allocator(
        cuda_memory_resource &resource,
        std::size_t minimum_size, 
        std::size_t request_size_step
    );
    cuda_memory_block_allocator(const cuda_memory_block_allocator &other) = delete;
    cuda_memory_block_allocator(cuda_memory_block_allocator &&other) = default;
    ~cuda_memory_block_allocator() = default;

    cuda_memory_block_allocator&
    operator=(const cuda_memory_block_allocator &other) = delete;
    cuda_memory_block_allocator&
    operator=(cuda_memory_block_allocator &&other) = default;

    /**
     * @brief Get the memory resource used for allocation.
     * 
     * @return cuda_memory_resource& The memory resource used for allocation.
     */
    cuda_memory_resource& get_memory_resource() const noexcept;

    /**
     * @brief Return free blocks to the allocator when possible.
     * 
     */
    void release();

    /**
     * @brief Allocate a new block.
     * 
     * @param size Size of the requested block.
     * @param alignment Alignment requirement for the requested block.
     * @param queue Queue of the requested block.
     * @param usage_tracker Output parameter to register alien queues. May be 
     * nullptr. Ownership is managed by the allocator and the caller shall not
     * call any delete/free on it.
     * @return const cuda_memory_block& Suitable block.
     * 
     * @throws std::bad_alloc if no suitable memory block is available.
     * 
     */
    const cuda_memory_block& allocate(
        std::size_t size, 
        std::size_t alignment, 
        const cuda_device_queue *queue,
        cuda_memory_block_usage_tracker **usage_tracker 
    );

    /**
     * @brief Allocate a new block.
     * 
     * @param size Size of the requested block.
     * @param alignment Alignment requirement for the requested block.
     * @param queue Queue of the requested block.
     * @param usage_tracker Output parameter to register alien queues. May be 
     * nullptr. Ownership is managed by the allocator and the caller shall not
     * call any delete/free on it.
     * @return const cuda_memory_block* Suitable block. nullptr if the 
     * allocation failed.
     * 
     * @throws std::bad_alloc if no suitable memory block is available.
     * 
     */
    const cuda_memory_block* try_allocate(
        std::size_t size, 
        std::size_t alignment, 
        const cuda_device_queue *queue,
        cuda_memory_block_usage_tracker **usage_tracker 
    );

    /**
     * @brief Deallocate a block.
     * 
     * @param block Block to be deallocated.
     * 
     * @note This operation does not return the block to the allocator.
     * Instead, it caches it for potential re-use.
     * 
     */
    void deallocate(const cuda_memory_block &block);

private:
    std::reference_wrapper<cuda_memory_resource> m_resource;
    cuda_memory_block_pool m_block_pool;
    cuda_memory_block_deferred_release m_deferred_blocks;
    std::size_t m_minimum_size;
    std::size_t m_request_size_step;

}; 

} // namespace hardware
} // namespace xmipp4
