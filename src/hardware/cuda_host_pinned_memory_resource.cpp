// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_host_pinned_memory_resource.hpp"

#include <xmipp4/core/memory/align.hpp>
#include <xmipp4/cuda/hardware/cuda_error.hpp>

#include <utility>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace hardware
{

cuda_host_pinned_memory_resource cuda_host_pinned_memory_resource::m_instance;

device* cuda_host_pinned_memory_resource::get_target_device() const noexcept
{
    return nullptr;
}

memory_resource_kind cuda_host_pinned_memory_resource::get_kind() const noexcept
{
    return memory_resource_kind::host_staging;
}

std::shared_ptr<memory_heap> 
cuda_host_pinned_memory_resource::create_memory_heap()
{
    return nullptr; // TODO
}

} // namespace hardware
} // namespace xmipp4
