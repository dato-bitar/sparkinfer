#pragma once
// Device memory whose contents can leave the GPU and come back at the same addresses.
//
// A speculative draft holds ~2.3 GB of weights, quantized copies and operands that only a
// speculating request reads. Above the speculation group size nothing does, and on a 32 GB card
// that is the headroom concurrent serving needs for its prefill scratch and decode graphs. This
// arena lets the draft step aside: offload() copies every mapped byte to pinned host memory and
// returns the physical memory to the device; restore() maps new physical memory at the SAME
// virtual addresses and copies it back. Every pointer the draft holds (into its weights, its
// scratch, the slot-0 caches) stays valid across the round trip, so nothing has to be rebased.
//
// Built on the CUDA virtual memory API (cuMemAddressReserve / cuMemCreate / cuMemMap). Where that
// is unavailable, init() fails and the owner keeps plain cudaMalloc, without offload.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace sparkinfer {

class VmmArena {
public:
    VmmArena() = default;
    VmmArena(const VmmArena&) = delete;
    VmmArena& operator=(const VmmArena&) = delete;
    ~VmmArena();

    // Reserves va_bytes of address space on `device`. False if the virtual memory API is
    // unavailable; the arena is then unusable and alloc() returns nullptr.
    bool init(int device, size_t va_bytes);
    bool ready() const { return base_ != 0; }

    // nullptr when the device is out of memory (cudaMalloc semantics, without the sticky error).
    // Allocations up to a quarter of the granularity share chunks; larger ones get their own
    // physical memory, which free() returns.
    void* alloc(size_t bytes);
    // True if p came from this arena. Small allocations are kept until destruction.
    bool free(void* p);

    // Copies every mapped byte to pinned host memory and releases the physical memory. Returns
    // the bytes released (0 if already offloaded, empty, or the host buffer could not be pinned).
    // The caller guarantees nothing on the device reads or writes the arena meanwhile.
    size_t offload();
    // Maps the memory back at the same addresses and restores its contents. False (and still
    // offloaded) if the device cannot provide it.
    bool restore();
    bool offloaded() const { return offloaded_; }
    size_t mapped_bytes() const;
    // What restore() needs: every live block, mapped or not.
    size_t live_bytes() const;

private:
    struct Block {
        CUdeviceptr va = 0;
        size_t size = 0;
        CUmemGenericAllocationHandle h = 0;
        bool mapped = false;
        bool live = true;     // false once free()d: its address range is never reused
        bool small = false;   // a shared chunk of small allocations
        bool in_host = false; // its bytes are in the host copy, at host_off
        size_t host_off = 0;
    };
    bool map_block(Block& b);
    void unmap_block(Block& b);

    int dev_ = 0;
    CUdeviceptr base_ = 0;
    size_t va_size_ = 0, va_top_ = 0, gran_ = 0;
    std::vector<Block> blocks_;
    CUdeviceptr small_cur_ = 0;
    size_t small_left_ = 0;
    void* host_ = nullptr;
    size_t host_size_ = 0;
    bool offloaded_ = false;
    bool pin_failed_ = false;   // the host copy could not be pinned: do not retry on every pressure
};

}  // namespace sparkinfer
