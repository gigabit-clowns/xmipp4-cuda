// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/device.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_device final
    : public device
{
public:
    explicit cuda_device(int device);
    cuda_device(const cuda_device &other) = delete;
    cuda_device(cuda_device &&other) = delete;
    ~cuda_device() override = default;

    cuda_device& operator=(const cuda_device &other) = delete;
    cuda_device& operator=(cuda_device &&other) = delete;

    int get_index() const noexcept;

    void enumerate_memory_resources(
        std::vector<memory_resource*> &resources
    ) override;
    
    std::shared_ptr<device_queue>
    create_device_queue() override;

    std::shared_ptr<device_event> create_device_event() override;

    std::shared_ptr<device_to_host_event> 
    create_device_to_host_event() override;

private:
    int m_device;

}; 

} // namespace hardware
} // namespace xmipp4
