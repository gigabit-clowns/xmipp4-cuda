// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_device_memory_resource.hpp"

#include <xmipp4/cuda/hardware/cuda_device.hpp>
#include <xmipp4/cuda/hardware/cuda_error.hpp>

#include <xmipp4/core/memory/align.hpp>

#include "cuda_device_memory_allocator.hpp"

#include <utility>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace hardware
{

cuda_device_memory_resource::cuda_device_memory_resource(
    cuda_device &device
) noexcept
    : m_device(device)
{
}

device* cuda_device_memory_resource::get_target_device() const noexcept
{
    cuda_device &device = m_device.get();
    return &device;
}

memory_resource_kind cuda_device_memory_resource::get_kind() const noexcept
{
    return memory_resource_kind::device_local;
}

std::shared_ptr<memory_allocator> 
cuda_device_memory_resource::create_allocator()
{
    return std::make_shared<cuda_device_memory_allocator>(*this);
}

void* cuda_device_memory_resource::malloc(
    std::size_t size, 
    std::size_t alignment
)
{
    void* result;
    const auto device_id = m_device.get().get_index();
    XMIPP4_CUDA_CHECK( cudaSetDevice(device_id) );
    XMIPP4_CUDA_CHECK( cudaMalloc(&result, size) );

    if (!memory::is_aligned(result, alignment))
    {
        XMIPP4_CUDA_CHECK( cudaFree(result) );
        result = nullptr;
    }

    return result;
}

void cuda_device_memory_resource::free(void* ptr)
{
    XMIPP4_CUDA_CHECK( cudaFree(ptr) );
}

} // namespace hardware
} // namespace xmipp4
