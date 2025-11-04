// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_host_pinned_memory_allocator.hpp"

#include "cuda_host_pinned_memory_resource.hpp"
#include "cuda_host_pinned_buffer.hpp"

namespace xmipp4
{
namespace hardware
{

XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_HOST_MEMORY_REQUEST_ROUND_STEP = 512;
XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_HOST_MEMORY_ALLOCATE_ROUND_STEP = 2<<20; // 2MB

cuda_host_pinned_memory_allocator::cuda_host_pinned_memory_allocator()
    : m_allocator(
        cuda_host_pinned_memory_resource::get(),
        XMIPP4_CUDA_HOST_MEMORY_REQUEST_ROUND_STEP,
        XMIPP4_CUDA_HOST_MEMORY_ALLOCATE_ROUND_STEP
    )
{
}

cuda_memory_resource& 
cuda_host_pinned_memory_allocator::get_memory_resource() const noexcept
{
        cuda_host_pinned_memory_resource::get();
}

std::shared_ptr<cuda_buffer> cuda_host_pinned_memory_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    cuda_device_queue *queue
)
{
    return std::make_shared<cuda_host_pinned_buffer>(
        size, 
        alignment,
        queue,
        m_allocator
    );
}

std::shared_ptr<buffer> cuda_host_pinned_memory_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    device_queue *queue
)
{
    // TODO
}

} // namespace hardware
} // namespace xmipp4
