from __future__ import annotations

import struct
import numpy as np
from multiprocessing import shared_memory


class SharedEntityBuffer:
    MAX_ENTITIES = 15000

    _FIELD_DEFS = [
        ("x", np.float32),
        ("y", np.float32),
        ("dx", np.float32),
        ("dy", np.float32),
        ("energy", np.float32),
        ("max_energy", np.float32),
        ("health", np.float32),
        ("max_health", np.float32),
        ("age", np.int32),
        ("max_age", np.int32),
        ("metabolism", np.float32),
        ("size_gene", np.float32),
        ("speed_gene", np.float32),
        ("vision", np.float32),
        ("diet_type", np.int8),
        ("efficiency", np.float32),
        ("repro_threshold", np.float32),
        ("repro_cooldown", np.int32),
        ("repro_max_cooldown", np.int32),
        ("aggression", np.float32),
        ("alive", np.bool_),
        ("r", np.uint8),
        ("g", np.uint8),
        ("b", np.uint8),
        ("repro_type", np.int8),
        ("habitat", np.int8),
        ("speed_mod", np.float32),
        ("metabolism_mod", np.float32),
        ("efficiency_mod", np.float32),
        ("origin", np.int8),
        ("parent_eid", np.int32),
        ("eid", np.int32),
        ("photosynth", np.float32),
    ]

    def __init__(self, name: str | None = None, create: bool = True) -> None:
        self._offsets: dict[str, int] = {}
        self._dtypes: dict[str, np.dtype] = {}

        entity_bytes = 0
        for field_name, dtype in self._FIELD_DEFS:
            self._offsets[field_name] = entity_bytes
            self._dtypes[field_name] = np.dtype(dtype)
            entity_bytes += self.MAX_ENTITIES * self._dtypes[field_name].itemsize

        self._count_offset = entity_bytes
        self._tick_offset = self._count_offset + 4
        self._total_size = self._tick_offset + 4

        if create:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)

        self._buf = self._shm.buf

    def _get_array(self, field_name: str) -> np.ndarray:
        offset = self._offsets[field_name]
        dtype = self._dtypes[field_name]
        count = self.count
        return np.ndarray(
            count, dtype=dtype, buffer=self._buf, offset=offset
        )

    @property
    def count(self) -> int:
        return struct.unpack_from("i", self._buf, self._count_offset)[0]

    @count.setter
    def count(self, value: int) -> None:
        struct.pack_into("i", self._buf, self._count_offset, value)

    @property
    def tick(self) -> int:
        return struct.unpack_from("i", self._buf, self._tick_offset)[0]

    @tick.setter
    def tick(self, value: int) -> None:
        struct.pack_into("i", self._buf, self._tick_offset, value)

    def write_from_entity_data(self, ed) -> None:
        self.count = ed.count
        for field_name, _ in self._FIELD_DEFS:
            src = getattr(ed, field_name)
            dst = self._get_array(field_name)
            np.copyto(dst, src[:ed.count])

    def read_into_arrays(self) -> dict[str, np.ndarray]:
        n = self.count
        result = {}
        for field_name, _ in self._FIELD_DEFS:
            result[field_name] = self._get_array(field_name).copy()
        result["count"] = n
        result["tick"] = self.tick
        return result

    @property
    def shm_name(self) -> str:
        return self._shm.name

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.close()
        self._shm.unlink()
