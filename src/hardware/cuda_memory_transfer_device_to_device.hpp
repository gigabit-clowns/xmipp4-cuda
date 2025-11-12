// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_transfer.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_device_queue;
class cuda_buffer;

class cuda_memory_transfer_device_to_device
    : public memory_transfer
{
public:
    cuda_memory_transfer_device_to_device() = default;
    ~cuda_memory_transfer_device_to_device() override = default;

    void copy(
        const buffer &source, 
        buffer &destination,
        span<const copy_region> regions, 
        device_queue *queue
    ) const override;
    
    void copy(
        const cuda_buffer &source, 
        cuda_buffer &destination,
        span<const copy_region> regions, 
        cuda_device_queue *queue
    ) const;

private:
    int m_direction;

};

} // namespace hardware
} // namespace xmipp4
