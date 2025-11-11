// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/buffer.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_buffer
    : public buffer
{
public:
    cuda_buffer(
        void *device_pointer,
        void *host_pointer,
        std::size_t size,
        std::reference_wrapper<memory_resource> resource,
        std::unique_ptr<buffer_sentinel> sentinel
    );

    void* get_device_ptr() noexcept;

    const void* get_device_ptr() const noexcept;

private:
    void *m_device_ptr;

}; 

} // namespace hardware
} // namespace xmipp4
