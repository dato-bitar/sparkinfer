#include "vmm_arena.h"

#include <cstdio>

namespace sparkinfer {

namespace {
// The driver API, resolved through the runtime rather than linked: linking libcuda would make the
// runtime library refuse to load on a host without the driver (CI runners, `--help`), where it
// should start and report that there is no device. decltype keeps each pointer's exact type.
struct Driver {
    decltype(&::cuInit) Init = nullptr;
    decltype(&::cuDeviceGet) DeviceGet = nullptr;
    decltype(&::cuDeviceGetAttribute) DeviceGetAttribute = nullptr;
    decltype(&::cuMemGetAllocationGranularity) MemGetAllocationGranularity = nullptr;
    decltype(&::cuMemAddressReserve) MemAddressReserve = nullptr;
    decltype(&::cuMemAddressFree) MemAddressFree = nullptr;
    decltype(&::cuMemCreate) MemCreate = nullptr;
    decltype(&::cuMemRelease) MemRelease = nullptr;
    decltype(&::cuMemMap) MemMap = nullptr;
    decltype(&::cuMemUnmap) MemUnmap = nullptr;
    decltype(&::cuMemSetAccess) MemSetAccess = nullptr;
    bool ok = false;
};

template <class F> bool resolve(F& fn, const char* name) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || !p) {
        cudaGetLastError();
        return false;
    }
    fn = reinterpret_cast<F>(p);
    return true;
}

const Driver& driver() {
    static const Driver d = [] {
        Driver r;
        r.ok = resolve(r.Init, "cuInit") && resolve(r.DeviceGet, "cuDeviceGet") &&
               resolve(r.DeviceGetAttribute, "cuDeviceGetAttribute") &&
               resolve(r.MemGetAllocationGranularity, "cuMemGetAllocationGranularity") &&
               resolve(r.MemAddressReserve, "cuMemAddressReserve") &&
               resolve(r.MemAddressFree, "cuMemAddressFree") && resolve(r.MemCreate, "cuMemCreate") &&
               resolve(r.MemRelease, "cuMemRelease") && resolve(r.MemMap, "cuMemMap") &&
               resolve(r.MemUnmap, "cuMemUnmap") && resolve(r.MemSetAccess, "cuMemSetAccess");
        return r;
    }();
    return d;
}

CUmemAllocationProp device_prop(int dev) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}
size_t round_up(size_t n, size_t m) { return (n + m - 1) / m * m; }
}  // namespace

bool VmmArena::init(int device, size_t va_bytes) {
    // The runtime has made the primary context current by now; the driver API needs it initialized.
    const Driver& d0 = driver();
    if (!d0.ok || d0.Init(0) != CUDA_SUCCESS) return false;
    int vmm = 0;
    CUdevice d = 0;
    if (driver().DeviceGet(&d, device) != CUDA_SUCCESS ||
        driver().DeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, d) !=
            CUDA_SUCCESS ||
        !vmm)
        return false;
    const CUmemAllocationProp prop = device_prop(device);
    size_t g = 0;
    if (driver().MemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
            CUDA_SUCCESS || g == 0)
        return false;
    const size_t va = round_up(va_bytes, g);
    CUdeviceptr base = 0;
    if (driver().MemAddressReserve(&base, va, 0, 0, 0) != CUDA_SUCCESS) return false;
    dev_ = device;
    gran_ = g;
    base_ = base;
    va_size_ = va;
    return true;
}

VmmArena::~VmmArena() {
    if (!base_) return;
    cudaDeviceSynchronize();
    for (Block& b : blocks_) unmap_block(b);
    driver().MemAddressFree(base_, va_size_);
    if (host_) cudaFreeHost(host_);
}

bool VmmArena::map_block(Block& b) {
    const CUmemAllocationProp prop = device_prop(dev_);
    if (driver().MemCreate(&b.h, b.size, &prop, 0) != CUDA_SUCCESS) return false;
    if (driver().MemMap(b.va, b.size, 0, b.h, 0) != CUDA_SUCCESS) {
        driver().MemRelease(b.h);
        return false;
    }
    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = dev_;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (driver().MemSetAccess(b.va, b.size, &acc, 1) != CUDA_SUCCESS) {
        driver().MemUnmap(b.va, b.size);
        driver().MemRelease(b.h);
        return false;
    }
    b.mapped = true;
    return true;
}

void VmmArena::unmap_block(Block& b) {
    if (!b.mapped) return;
    driver().MemUnmap(b.va, b.size);
    driver().MemRelease(b.h);
    b.mapped = false;
}

void* VmmArena::alloc(size_t bytes) {
    if (!base_ || offloaded_) return nullptr;
    bytes = round_up(bytes ? bytes : 1, 256);
    if (bytes <= gran_ / 4) {
        if (small_left_ < bytes) {
            if (va_top_ + gran_ > va_size_) return nullptr;
            Block b;
            b.va = base_ + va_top_;
            b.size = gran_;
            b.small = true;
            if (!map_block(b)) return nullptr;
            va_top_ += gran_;
            blocks_.push_back(b);
            small_cur_ = b.va;
            small_left_ = gran_;
        }
        void* p = reinterpret_cast<void*>(small_cur_);
        small_cur_ += bytes;
        small_left_ -= bytes;
        return p;
    }
    const size_t size = round_up(bytes, gran_);
    if (va_top_ + size > va_size_) return nullptr;
    Block b;
    b.va = base_ + va_top_;
    b.size = size;
    if (!map_block(b)) return nullptr;
    va_top_ += size;
    blocks_.push_back(b);
    return reinterpret_cast<void*>(b.va);
}

bool VmmArena::free(void* p) {
    const CUdeviceptr va = reinterpret_cast<CUdeviceptr>(p);
    if (!base_ || va < base_ || va >= base_ + va_size_) return false;
    for (Block& b : blocks_) {
        if (b.small || !b.live || b.va != va) continue;
        cudaDeviceSynchronize();   // as cudaFree would: nothing may still be reading it
        unmap_block(b);
        b.live = false;
        return true;
    }
    return true;   // a small allocation: its chunk stays until the arena goes
}

size_t VmmArena::mapped_bytes() const {
    size_t n = 0;
    for (const Block& b : blocks_)
        if (b.mapped) n += b.size;
    return n;
}

size_t VmmArena::live_bytes() const {
    size_t n = 0;
    for (const Block& b : blocks_)
        if (b.live) n += b.size;
    return n;
}

size_t VmmArena::offload() {
    if (!base_ || offloaded_ || pin_failed_) return 0;
    const size_t total = mapped_bytes();
    if (total == 0) return 0;
    if (host_size_ < total) {
        if (host_) cudaFreeHost(host_);
        host_ = nullptr;
        host_size_ = 0;
        // Pinning stalls every thread's CUDA calls for its duration, so take a quarter more than
        // today's need: buffers the draft builds on first use must not force a second pinning.
        const size_t want = total + total / 4;
        if (cudaHostAlloc(&host_, want, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            host_ = nullptr;
            pin_failed_ = true;
            fprintf(stderr, "[vmm] cannot pin %.2f GB of host memory: the draft stays on the device\n",
                    (double)want / 1e9);
            return 0;
        }
        host_size_ = want;
    }
    cudaDeviceSynchronize();
    size_t off = 0;
    for (Block& b : blocks_) {
        b.in_host = false;
        if (!b.mapped) continue;
        if (cudaMemcpy(static_cast<char*>(host_) + off, reinterpret_cast<void*>(b.va), b.size,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            return 0;   // nothing released yet: still resident and intact
        }
        // Each block keeps its own offset, so a free() while offloaded cannot shift the others'.
        b.in_host = true;
        b.host_off = off;
        off += b.size;
    }
    for (Block& b : blocks_)
        if (b.live) unmap_block(b);
    offloaded_ = true;
    return total;
}

bool VmmArena::restore() {
    if (!offloaded_) return true;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        Block& b = blocks_[i];
        if (!b.live || !b.in_host) continue;
        if (!map_block(b)) {
            for (size_t j = 0; j < i; ++j) unmap_block(blocks_[j]);
            return false;
        }
        if (cudaMemcpy(reinterpret_cast<void*>(b.va), static_cast<char*>(host_) + b.host_off, b.size,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            for (size_t j = 0; j <= i; ++j) unmap_block(blocks_[j]);
            return false;
        }
    }
    offloaded_ = false;
    return true;
}

}  // namespace sparkinfer
