// Small CUDA Driver API bridge. No kernels, allocator replacement, or global
// configuration changes. The caller retains each allocation until GPU idle.
#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <limits>

extern "C" int iss_vmm_alloc(uint64_t bytes, int device, int compression,
                             uint64_t* pointer, uint64_t* handle,
                             uint64_t* allocated, int* actual_compression) {
    if (!bytes || !pointer || !handle || !allocated || !actual_compression)
        return CUDA_ERROR_INVALID_VALUE;
    *pointer = *handle = *allocated = 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.compressionType = compression ? CU_MEM_ALLOCATION_COMP_GENERIC : CU_MEM_ALLOCATION_COMP_NONE;
    size_t granularity = 0;
    CUresult status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status != CUDA_SUCCESS) return status;
    if (!granularity || bytes > std::numeric_limits<size_t>::max() - granularity + 1)
        return CUDA_ERROR_INVALID_VALUE;
    size_t size = (bytes + granularity - 1) / granularity * granularity;
    CUmemGenericAllocationHandle allocation{};
    status = cuMemCreate(&allocation, size, &prop, 0);
    if (status != CUDA_SUCCESS) return status;
    CUmemAllocationProp obtained{};
    status = cuMemGetAllocationPropertiesFromHandle(&obtained, allocation);
    if (status != CUDA_SUCCESS) { cuMemRelease(allocation); return status; }
    CUdeviceptr address{};
    status = cuMemAddressReserve(&address, size, 0, 0, 0);
    if (status != CUDA_SUCCESS) { cuMemRelease(allocation); return status; }
    status = cuMemMap(address, size, 0, allocation, 0);
    if (status != CUDA_SUCCESS) { cuMemAddressFree(address, size); cuMemRelease(allocation); return status; }
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status = cuMemSetAccess(address, size, &access, 1);
    if (status != CUDA_SUCCESS) {
        cuMemUnmap(address, size); cuMemAddressFree(address, size); cuMemRelease(allocation);
        return status;
    }
    *pointer = address; *handle = allocation; *allocated = size;
    *actual_compression = obtained.allocFlags.compressionType;
    return CUDA_SUCCESS;
}

extern "C" int iss_vmm_free(uint64_t pointer, uint64_t handle, uint64_t size) {
    CUresult status = cuMemUnmap(pointer, size);
    if (status != CUDA_SUCCESS) return status;
    status = cuMemAddressFree(pointer, size);
    CUresult released = cuMemRelease(handle);
    return status != CUDA_SUCCESS ? status : released;
}
