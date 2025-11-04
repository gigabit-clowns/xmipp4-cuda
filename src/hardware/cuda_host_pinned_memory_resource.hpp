// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_host_pinned_memory_resource final
    : public cuda_memory_resource
{
public:
    device* get_target_device() const noexcept override;

    memory_resource_kind get_kind() const noexcept override;

    std::shared_ptr<memory_allocator> create_allocator() override;

    void* malloc(std::size_t size, std::size_t alignment) override;

    void free(void* ptr) override;

    static cuda_host_pinned_memory_resource& get() noexcept;

private:    
    static cuda_host_pinned_memory_resource m_instance;

}; 

} // namespace hardware
} // namespace xmipp4

