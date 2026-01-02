import torch
from typing import Optional


class PartitionStage:
    """Container for staged partition state."""

    def __init__(self, partition_id: int, version: int, device_view: torch.Tensor, host_view: torch.Tensor):
        self.partition_id = partition_id
        self.version = version
        self.num_triples = device_view.size(0)
        self.device_view = device_view
        self.host_view = host_view


class PartitionStager:
    """Stages partition triples in pinned host memory and on a target device."""

    def __init__(self, device: torch.device, triple_dim: int = 3):
        self.device = device
        self.triple_dim = triple_dim
        self._device_buffer: Optional[torch.Tensor] = None
        self._host_buffer: Optional[torch.Tensor] = None
        self._capacity = 0
        self._current: Optional[PartitionStage] = None

    def _ensure_capacity(self, size: int, dtype: torch.dtype):
        if size <= self._capacity and self._device_buffer is not None:
            return
        self._capacity = size
        self._host_buffer = torch.empty((size, self.triple_dim), dtype=dtype, pin_memory=True)
        self._device_buffer = torch.empty((size, self.triple_dim), dtype=dtype, device=self.device)

    def stage(self, partition_id: int, version: int, triples: torch.Tensor) -> PartitionStage:
        if triples.dim() != 2 or triples.size(1) != self.triple_dim:
            raise ValueError("Expected triples shaped [N, triple_dim]")
        num_triples = triples.size(0)
        self._ensure_capacity(num_triples, triples.dtype)
        assert self._host_buffer is not None and self._device_buffer is not None
        self._host_buffer[:num_triples].copy_(triples, non_blocking=False)
        device_view = self._device_buffer[:num_triples]
        device_view.copy_(triples.to(self.device), non_blocking=True)
        host_view = self._host_buffer[:num_triples]
        self._current = PartitionStage(partition_id, version, device_view, host_view)
        return self._current

    def current(self) -> Optional[PartitionStage]:
        return self._current
