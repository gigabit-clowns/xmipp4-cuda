// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_device_memory_resource.hpp"

#include <xmipp4/cuda/hardware/cuda_device.hpp>
#include <xmipp4/cuda/hardware/cuda_error.hpp>

#include <xmipp4/core/memory/align.hpp>

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

std::shared_ptr<memory_heap> cuda_device_memory_resource::create_memory_heap()
{
    return nullptr; // TODO
}

} // namespace hardware
} // namespace xmipp4
