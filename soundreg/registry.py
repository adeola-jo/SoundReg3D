"""Tiny name -> class registries so components are swappable from config."""

from __future__ import annotations


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._members: dict[str, type] = {}

    def register(self, name: str):
        def deco(cls):
            if name in self._members:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._members[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type:
        if name not in self._members:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._members)}"
            )
        return self._members[name]

    def names(self) -> list[str]:
        return sorted(self._members)


ENCODERS = Registry("encoder")
MODELS = Registry("model")
ORDERINGS = Registry("ordering")
