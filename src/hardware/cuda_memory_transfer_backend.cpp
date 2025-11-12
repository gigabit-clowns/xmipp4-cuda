// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_transfer_backend.hpp"

#include <xmipp4/core/hardware/memory_transfer_manager.hpp>

//#include "cuda_memory_transfer.hpp"

namespace xmipp4
{
namespace hardware
{
backend_priority cuda_memory_transfer_backend::get_suitability(
    const memory_resource& source,
    const memory_resource& destination
) const noexcept
{
    return backend_priority::unsupported; // TODO
}

std::shared_ptr<memory_transfer> 
cuda_memory_transfer_backend::create_transfer(
    const memory_resource& src,
    const memory_resource& dst
) const
{
    return nullptr; // TODO
}

bool cuda_memory_transfer_backend::register_at(memory_transfer_manager &manager)
{
    return manager.register_backend(
        std::make_unique<cuda_memory_transfer_backend>()
    );
}

} // namespace hardware
} // namespace xmipp4
