// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

#include <utility>

namespace xmipp4 
{
namespace hardware
{

class cuda_device;

class cuda_device_memory_resource final
    : public cuda_memory_resource
{
public:
    explicit cuda_device_memory_resource(cuda_device &device) noexcept;
    cuda_device_memory_resource(const cuda_device_memory_resource &other) = default;
    cuda_device_memory_resource(cuda_device_memory_resource &&other) = default;
    ~cuda_device_memory_resource() override = default;

    cuda_device_memory_resource&
    operator=(const cuda_device_memory_resource &other) = default;
    cuda_device_memory_resource&
    operator=(cuda_device_memory_resource &&other) = default;

    device* get_target_device() const noexcept override;

    memory_resource_kind get_kind() const noexcept override;

    std::shared_ptr<memory_allocator> create_allocator() override;

    void* malloc(std::size_t size, std::size_t alignment) override;

    void free(void* ptr) override;

private:
    std::reference_wrapper<cuda_device> m_device;

}; 

} // namespace hardware
} // namespace xmipp4

