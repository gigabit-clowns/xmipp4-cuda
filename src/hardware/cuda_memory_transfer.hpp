// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_transfer.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_device_queue;

class cuda_memory_transfer
    : public memory_transfer
{
public:
    explicit cuda_memory_transfer(int direction) noexcept;
    cuda_memory_transfer(const cuda_memory_transfer &other) = default;
    cuda_memory_transfer(cuda_memory_transfer &&other) = default;
    ~cuda_memory_transfer() override = default;

    cuda_memory_transfer&
    operator=(const cuda_memory_transfer &other) = default;
    cuda_memory_transfer&
    operator=(cuda_memory_transfer &&other) = default;

    void copy(
        const buffer &source, 
        buffer &destination,
        span<const copy_region> regions, 
        device_queue *queue
    ) const override;
    
    void copy(
        const buffer &source, 
        buffer &destination,
        span<const copy_region> regions, 
        cuda_device_queue *queue
    ) const;

private:
    int m_direction;

};

} // namespace hardware
} // namespace xmipp4
