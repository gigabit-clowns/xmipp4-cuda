// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_copy_bytes.hpp"

#include <xmipp4/core/hardware/copy_region.hpp>
#include <xmipp4/core/memory/align.hpp>

#include <stdexcept>

namespace xmipp4 
{
namespace hardware
{

inline
void cuda_copy_bytes(
    const void *src_ptr, 
    std::size_t src_size, 
    void* dst_ptr, 
    std::size_t dst_size,
    const copy_region &region,
    cudaMemcpyKind direction,
    cudaStream_t stream
)
{
    const auto src_offset = region.get_source_offset();
    const auto dst_offset = region.get_destination_offset();
    const auto byte_count = region.get_count();

    if (src_offset + byte_count > src_size)
    {
        throw std::out_of_range(
            "Copy region exceeds source buffer size."
        );
    }

    if (dst_offset + byte_count > dst_size)
    {
        throw std::out_of_range(
            "Copy region exceeds destination buffer size."
        );
    }

    cudaMemcpyAsync(
        memory::offset_bytes(dst_ptr, dst_offset), 
        memory::offset_bytes(src_ptr, src_offset), 
        byte_count,
        direction,
        stream
    );
}

} // namespace hardware
} // namespace xmipp4
