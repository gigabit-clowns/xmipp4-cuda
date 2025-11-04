// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_resource.hpp>

#include <cstddef>

namespace xmipp4 
{
namespace hardware
{

class cuda_memory_allocator;

class cuda_memory_resource
    : public memory_resource
{
public:
    virtual void* malloc(std::size_t size, std::size_t alignment) = 0;
    virtual void free(void* ptr) = 0;

    std::shared_ptr<cuda_memory_allocator> create_cuda_allocator();
    std::shared_ptr<memory_allocator> create_allocator() override;

}; 

} // namespace hardware
} // namespace xmipp4
