from __future__ import annotations

from typing import TypeVar, Type, Iterator

T = TypeVar("T")


class Component:
    pass


class EntityManager:
    def __init__(self) -> None:
        self._next_id: int = 0
        self._components: dict[Type[Component], dict[int, Component]] = {}
        self._entities: set[int] = set()
        self._query_cache: dict[tuple[str, ...], frozenset] = {}
        self._cache_dirty: bool = False

    def _invalidate_cache(self) -> None:
        self._cache_dirty = True

    def create_entity(self, *components: Component) -> int:
        eid = self._next_id
        self._next_id += 1
        self._entities.add(eid)
        for comp in components:
            self.add_component(eid, comp)
        self._invalidate_cache()
        return eid

    def remove_entity(self, entity_id: int) -> None:
        if entity_id not in self._entities:
            return
        for comp_type in list(self._components.keys()):
            self._components[comp_type].pop(entity_id, None)
        self._entities.discard(entity_id)
        self._invalidate_cache()

    def add_component(self, entity_id: int, component: Component) -> None:
        comp_type = type(component)
        if comp_type not in self._components:
            self._components[comp_type] = {}
        self._components[comp_type][entity_id] = component

    def get_component(self, entity_id: int, comp_type: Type[T]) -> T | None:
        bucket = self._components.get(comp_type)
        if bucket is None:
            return None
        return bucket.get(entity_id)

    def has_component(self, entity_id: int, comp_type: Type[Component]) -> bool:
        bucket = self._components.get(comp_type)
        if bucket is None:
            return False
        return entity_id in bucket

    def get_entities_with(self, *comp_types: Type[Component]) -> Iterator[int]:
        if not comp_types:
            return
        key = tuple(ct.__name__ for ct in comp_types)
        if self._cache_dirty:
            self._query_cache.clear()
            self._cache_dirty = False
        elif key in self._query_cache:
            yield from self._query_cache[key]
            return

        sets = []
        for ct in comp_types:
            bucket = self._components.get(ct)
            if bucket is None:
                self._query_cache[key] = frozenset()
                return
            sets.append(bucket.keys())
        common = frozenset(sets[0])
        for s in sets[1:]:
            common = common.intersection(s)
        self._query_cache[key] = common
        yield from common

    def get_all_of_type(self, comp_type: Type[T]) -> dict[int, T]:
        return dict(self._components.get(comp_type, {}))

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def all_entities(self) -> set[int]:
        return set(self._entities)


class System:
    def update(self, world: object, dt: float) -> None:
        raise NotImplementedError
