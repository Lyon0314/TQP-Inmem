import torch
import constants
from utility.tensor_utils import dtype_size
from utility.logger import message_logger
from typing_extensions import List
import threading
from utility.logger import message_logger
from utility.tensor_utils import dtype_size

class MemoryPool:
    COALESCE_THRESHOLD = 5
    
    def __init__(self, capacity_gb: int, block_size_mb: int):
        self.pool_size_bytes = int(capacity_gb * 1000**3)
        self.block_size_bytes = int(block_size_mb * 1000**2)
        
        if self.pool_size_bytes < self.block_size_bytes:
            raise ValueError("Pool size must be larger than the block size.")

        self._num_blocks = self.pool_size_bytes // self.block_size_bytes
        self._free = [(0, self._num_blocks)]
        self._allocs = {}
        self._free_times = 0

    def get_storage(self, actual_size_bytes):
        alloc_blocks = max((actual_size_bytes + self.block_size_bytes - 1) // self.block_size_bytes, 1)
        # message_logger().debug(self.dump_free())
        # message_logger().debug("view=%s size=%s block=%s", subject_tensor_view, actual_size_bytes, alloc_blocks)
        for i, (start, length) in enumerate(self._free):
            if length >= alloc_blocks:
                ptr = start 
                if length == alloc_blocks:
                    self._free.pop(i)
                else: 
                    self._free[i] = (start + alloc_blocks, length - alloc_blocks)
                self._allocs[ptr] = alloc_blocks
                return ptr, self._mem_pool[start * self.block_size_bytes : \
                                           start * self.block_size_bytes + actual_size_bytes]  # only return actual size for view
        raise MemoryError(f"Pinned memory cannot allocate {alloc_blocks} blocks")
    
    def malloc(self, num_elements, to_alloc_dtype):
        ptr, storage = self.get_storage(num_elements * dtype_size(to_alloc_dtype))
        return ptr, storage.view(to_alloc_dtype)
    
    def malloc_like(self, subject_tensor: torch.Tensor):
        ptr, storage = self.get_storage(subject_tensor.numel() * subject_tensor.element_size())
        return ptr, storage.view(subject_tensor.dtype).view(subject_tensor.shape) 
    
    def malloc_list(self, subject:List[torch.tensor]):
        ptrs, target = [], []
        for i in range(len(subject)):
            ptr, storage = self.malloc_like(subject[i])
            target.append(storage)
            ptrs.append(ptr)
        return ptrs, target
    
    def free_all(self):
        self._free = [(0, self._num_blocks)]
        self._allocs.clear()

    def free(self, ptr: int):
        length = self._allocs.pop(ptr, None)
        if length is None:
            raise ValueError(f"{ptr} not found in pool")

        self._free.append((ptr, length))
        
        self._free_times += 1
        if self._free_times > self.COALESCE_THRESHOLD:
            self.coalesce()
            
    def coalesce(self):
        self._free.sort(key=lambda x: x[0])
        coalesced = []
        cur_start, cur_len = self._free[0]
        for st, ln in self._free[1:]:
            if cur_start + cur_len == st:
                cur_len += ln
            else:
                coalesced.append((cur_start, cur_len))
                cur_start, cur_len = st, ln
        coalesced.append((cur_start, cur_len))
        self._free = coalesced

    def dump_free(self):
        """List of free (start_block, length_blocks)."""
        return list(self._free)
    
    def dump_allocs(self):
        """Map of allocated ptr_block -> length_blocks."""
        return dict(self._allocs)
    
    def free_space(self) -> int:
        """Return the total free space in bytes."""
        return max(length for _, length in self._free) * self.block_size_bytes

    def partial_free(self, ptr: int, bytes_to_retain: int):
        length = self._allocs.get(ptr, None)
        if length is None:
            raise ValueError(f"{ptr} not found in pool")
        block_size = self.block_size_bytes
        blocks_to_retain = (bytes_to_retain + block_size - 1) // block_size
        if blocks_to_retain >= length:
            # Retain all, nothing to free
            return
        
        blocks_to_free = length - blocks_to_retain
        if blocks_to_free == self._allocs[ptr]:
            self.free(ptr)
            return
   
        free_start = ptr + blocks_to_retain
        self._allocs[ptr] = blocks_to_retain
        self._free.append((free_start, blocks_to_free))
        self._free_times += 1
        if self._free_times > self.COALESCE_THRESHOLD:
            self.coalesce()
    
    def get_external_frag(self):
        return 1 - (max(x[1] for x in self._free) / sum(x[1] for x in self._free))
    
    def get_utilization(self):
        return sum(self._allocs.values()) / self._num_blocks
    
class PinnedMemory(MemoryPool):
    def __init__(self, capacity_gb: int, block_size_mb: int):
        super().__init__(capacity_gb, block_size_mb)
        self._mem_pool = torch.empty(self.pool_size_bytes, dtype=torch.uint8).pin_memory()
    

class GPUMemory(MemoryPool):
    DEVICE = 'cuda:0'

    def __init__(self, capacity_gb: int, block_size_mb: int):
        super().__init__(capacity_gb, block_size_mb)
        self._mem_pool = torch.empty(self.pool_size_bytes, dtype=torch.uint8, device=self.DEVICE)
