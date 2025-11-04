// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_allocator.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_buffer;
class cuda_device_queue;

class cuda_memory_allocator
    : public memory_allocator
{
public:
    virtual
    std::shared_ptr<cuda_buffer> allocate(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue
    ) = 0;

}; 

} // namespace hardware
} // namespace xmipp4
