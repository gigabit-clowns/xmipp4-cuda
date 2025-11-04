// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_device_memory_allocator.hpp"

#include "cuda_device_memory_resource.hpp"
#include "cuda_device_buffer.hpp"

namespace xmipp4
{
namespace hardware
{

XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_DEVICE_MEMORY_REQUEST_ROUND_STEP = 256;
XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_DEVICE_MEMORY_ALLOCATE_ROUND_STEP = 2<<20; // 2MB

cuda_device_memory_allocator::cuda_device_memory_allocator(
    cuda_device_memory_resource &resource
)
    : m_allocator(
        resource, 
        XMIPP4_CUDA_DEVICE_MEMORY_REQUEST_ROUND_STEP,
        XMIPP4_CUDA_DEVICE_MEMORY_ALLOCATE_ROUND_STEP
    )
{
}

cuda_memory_resource& 
cuda_device_memory_allocator::get_memory_resource() const noexcept
{
    return m_allocator.get_memory_resource();
}

std::shared_ptr<cuda_buffer> cuda_device_memory_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    cuda_device_queue *queue
)
{
    return std::make_shared<cuda_device_buffer>(
        size, 
        alignment,
        queue,
        m_allocator
    );
}

std::shared_ptr<buffer> cuda_device_memory_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    device_queue *queue
)
{
    // TODO
}

} // namespace hardware
} // namespace xmipp4
